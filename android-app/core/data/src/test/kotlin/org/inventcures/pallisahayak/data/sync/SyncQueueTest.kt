package org.inventcures.pallisahayak.data.sync

import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.inventcures.pallisahayak.data.repository.FakeApi
import org.inventcures.pallisahayak.data.repository.FakeInteractionDao
import org.inventcures.pallisahayak.data.repository.PalliSahayakRepository
import org.inventcures.pallisahayak.data.repository.SessionContext
import org.inventcures.pallisahayak.data.sync.wire.SyncAcknowledgement
import org.inventcures.pallisahayak.data.sync.wire.SyncApi
import org.inventcures.pallisahayak.data.sync.wire.SyncPushBody
import org.inventcures.pallisahayak.data.sync.wire.SyncPushResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import okhttp3.ResponseBody.Companion.toResponseBody
import retrofit2.Response
import java.io.IOException

/**
 * Offline observations sync in a background job and leave the queue only when
 * the server acknowledges each one.
 *
 * The contract is report 09's: one acknowledgement per pushed item, matched to
 * it by (kind, index), where index is the item's place in its kind's list.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class SyncQueueTest {

    private val dao = FakeSyncQueueDao()
    private val api = FakeSyncApi()
    private var now = 1_700_000_000_000L
    private val queue = SyncQueue(dao, api, clock = { now++ })

    private suspend fun observation(entity: String): String =
        queue.enqueueObservation(patientId = "patient-1", category = "symptom", entityName = entity, severity = 3)

    private fun ack(kind: String, index: Int, status: String, reason: String? = null) =
        SyncAcknowledgement(kind = kind, index = index, id = null, status = status, reason = reason)

    @Test
    fun `a sync with one accepted and one rejected item removes only the accepted one`() = runTest {
        val first = observation("pain")
        val second = observation("breathlessness")
        api.acks = listOf(ack("observation", 0, "accepted"), ack("observation", 1, "rejected", "save_failed"))

        val outcome = queue.push()

        assertEquals(listOf(second), dao.rows.map { it.id })
        assertEquals(1, outcome.accepted)
        assertEquals(1, outcome.retained)
        assertFalse(first in dao.rows.map { it.id })
    }

    @Test
    fun `the push carries each kind in queue order, with the item's id`() = runTest {
        val first = observation("pain")
        val second = observation("nausea")
        api.acks = emptyList()

        queue.push()

        val pushed = api.bodies.single().observations
        assertEquals(listOf(first, second), pushed.map { it["observation_id"] })
        assertEquals(listOf("pain", "nausea"), pushed.map { it["entity_name"] })
        assertEquals("patient-1", pushed.first()["patient_id"])
    }

    @Test
    fun `acknowledgements are matched on kind and index, not on their order`() = runTest {
        val obs = observation("pain")
        val log = queue.enqueue(SyncItem(SyncKind.INTERACTION_LOG, "log-1", mapOf("event_type" to "offline_interaction")))
        // Both items are index 0 of their own list; the log's ack comes first.
        api.acks = listOf(ack("interaction_log", 0, "rejected", "unavailable"), ack("observation", 0, "accepted"))

        queue.push()

        assertEquals(listOf(log), dao.rows.map { it.id })
        assertFalse(obs in dao.rows.map { it.id })
    }

    @Test
    fun `an item with no acknowledgement stays queued`() = runTest {
        val first = observation("pain")
        val second = observation("nausea")
        api.acks = listOf(ack("observation", 1, "accepted"))

        queue.push()

        assertEquals(listOf(first), dao.rows.map { it.id })
        assertFalse(second in dao.rows.map { it.id })
    }

    @Test
    fun `a push that fails on the network keeps every item`() = runTest {
        observation("pain")
        observation("nausea")
        api.offline = true

        val outcome = queue.push()

        assertEquals(2, dao.rows.size)
        assertFalse(outcome.reachedServer)
    }

    @Test
    fun `a push the server refuses whole keeps every item`() = runTest {
        observation("pain")
        api.status = 401

        queue.push()

        assertEquals(1, dao.rows.size)
    }

    @Test
    fun `an item the server will never accept is kept but not sent again`() = runTest {
        val refused = observation("pain")
        val retried = observation("nausea")
        api.acks = listOf(ack("observation", 0, "rejected", "not_owned"), ack("observation", 1, "rejected", "save_failed"))

        queue.push()
        api.acks = listOf(ack("observation", 0, "accepted"))
        queue.push()

        assertEquals(listOf(retried), api.bodies[1].observations.map { it["observation_id"] })
        assertEquals(listOf(refused), dao.rows.map { it.id })
        assertEquals("not_owned", dao.rows.single().refusedReason)
    }

    @Test
    fun `a push sends at most fifty items of a kind`() = runTest {
        repeat(51) { observation("pain $it") }
        api.acks = (0 until 50).map { ack("observation", it, "accepted") }

        queue.push()

        assertEquals(50, api.bodies.single().observations.size)
        assertEquals(1, dao.rows.size)
    }

    @Test
    fun `an empty queue makes no call`() = runTest {
        val outcome = queue.push()

        assertTrue(api.bodies.isEmpty())
        assertEquals(0, outcome.sent)
    }

    @Test
    fun `the background job pushes when the connection comes back`() = runTest {
        observation("pain")
        api.acks = listOf(ack("observation", 0, "accepted"))
        val online = MutableStateFlow(false)
        SyncScheduler(queue, online, backgroundScope, retryEveryMillis = 60_000).start()

        runCurrent()
        assertTrue("nothing is sent with no connection", api.bodies.isEmpty())

        online.value = true
        runCurrent()
        assertEquals(1, api.bodies.size)
        assertTrue(dao.rows.isEmpty())
    }

    @Test
    fun `the background job tries again later while items remain`() = runTest {
        observation("pain")
        api.acks = listOf(ack("observation", 0, "rejected", "save_failed"))
        SyncScheduler(queue, MutableStateFlow(true), backgroundScope, retryEveryMillis = 60_000).start()

        runCurrent()
        api.acks = listOf(ack("observation", 0, "accepted"))
        advanceTimeBy(60_001)

        assertEquals(2, api.bodies.size)
        assertTrue(dao.rows.isEmpty())
    }

    @Test
    fun `an offline answer is queued for sync as an interaction log, without its text`() = runTest {
        val api = FakeApi().apply { offline = true }
        val repository = PalliSahayakRepository(api = api, interactions = FakeInteractionDao(), outbox = queue)
        val session = SessionContext(
            participantId = "3f2a8c14-9b7e-4d21-8a55-6c1e0d3b7f92",
            careRole = "asha_worker",
            siteId = "CCHRC",
            language = "en-IN",
            releaseId = "study-1@abc1234",
            releaseApproved = true,
        )

        repository.ask("My mother Sita has pain, what can I do?", session)

        val item = dao.rows.single()
        assertEquals(SyncKind.INTERACTION_LOG.wire, item.kind)
        assertFalse("no question text leaves the phone", item.payloadJson.contains("Sita"))
        assertTrue(item.payloadJson.contains("\"is_offline\":true"))
        assertTrue(item.payloadJson.contains("\"log_id\":\"${item.id}\""))
        assertNull(item.refusedReason)
    }

    @Test
    fun `an answered online question is not queued`() = runTest {
        val api = FakeApi().apply { answer("Keep the patient comfortable and the room cool.") }
        val repository = PalliSahayakRepository(api = api, interactions = FakeInteractionDao(), outbox = queue)
        val session = SessionContext("p", "asha_worker", "CCHRC", "en-IN", "r", true)

        repository.ask("What helps with fever?", session)

        assertTrue(dao.rows.isEmpty())
    }
}

/** The sync queue table, in memory. */
class FakeSyncQueueDao : SyncQueueDao {
    val rows = mutableListOf<SyncItemEntity>()

    override suspend fun enqueue(item: SyncItemEntity) {
        rows.removeAll { it.id == item.id }
        rows.add(item)
    }

    override suspend fun pending(kind: String, limit: Int): List<SyncItemEntity> =
        rows.filter { it.kind == kind && it.refusedReason == null }
            .sortedWith(compareBy({ it.queuedAt }, { it.id }))
            .take(limit)

    override suspend fun remove(ids: List<String>) {
        rows.removeAll { it.id in ids }
    }

    override suspend fun refuse(id: String, reason: String) {
        val index = rows.indexOfFirst { it.id == id }
        if (index >= 0) rows[index] = rows[index].copy(refusedReason = reason)
    }

    override suspend fun pendingCount(): Int = rows.count { it.refusedReason == null }
}

/** The server's sync push, answering with [acks]. */
class FakeSyncApi : SyncApi {
    var offline = false
    var status = 200
    var acks: List<SyncAcknowledgement> = emptyList()
    val bodies = mutableListOf<SyncPushBody>()

    override suspend fun push(body: SyncPushBody): Response<SyncPushResult> {
        if (offline) throw IOException("no signal")
        bodies.add(body)
        if (status != 200) {
            return Response.error(status, "{\"status\":\"error\"}".toResponseBody())
        }
        val accepted = acks.count { it.status == "accepted" }
        return Response.success(SyncPushResult(accepted, acks.size - accepted, acks))
    }
}
