package org.inventcures.pallisahayak.data.sync

import android.util.Log
import androidx.room.ColumnInfo
import androidx.room.Dao
import androidx.room.Entity
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.PrimaryKey
import androidx.room.Query
import com.squareup.moshi.JsonDataException
import com.squareup.moshi.Moshi
import com.squareup.moshi.Types
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.inventcures.pallisahayak.data.local.InteractionEntity
import org.inventcures.pallisahayak.data.sync.wire.SyncApi
import org.inventcures.pallisahayak.data.sync.wire.SyncPushBody
import java.io.IOException
import java.util.UUID

/**
 * What can be pushed, by the server's names (`sync/batch_processor.py`): the
 * acknowledgement kind, and the field that carries the item's id.
 */
enum class SyncKind(val wire: String, val idField: String) {
    OBSERVATION("observation", "observation_id"),
    MEDICATION_REMINDER("medication_reminder", "reminder_id"),
    INTERACTION_LOG("interaction_log", "log_id"),
    VIGNETTE_RESPONSE("vignette_response", "response_id");

    companion object {
        fun fromWire(value: String?): SyncKind? = entries.firstOrNull { it.wire == value }
    }
}

/** One item to push: its kind, its id, and the fields the server reads. */
data class SyncItem(val kind: SyncKind, val id: String, val payload: Map<String, Any?>)

/**
 * Where something the server has not seen yet waits for a sync.
 *
 * An interface so the repository can queue without knowing about Room or the
 * network, and so a test can see what was queued.
 */
interface SyncOutbox {
    /** Queue [item]; returns its id. Queuing the same id again replaces it. */
    suspend fun enqueue(item: SyncItem): String
}

/**
 * One queued item. It leaves the table only when the server acknowledges it as
 * accepted. An item the server will never accept ("invalid", "not_owned") stays
 * on the phone with [refusedReason] set, for the study team to see, and is not
 * sent again.
 */
@Entity(tableName = "sync_queue")
data class SyncItemEntity(
    @PrimaryKey
    val id: String,

    /** [SyncKind.wire]. */
    val kind: String,

    /** The item as the server reads it, as JSON. */
    @ColumnInfo(name = "payload_json")
    val payloadJson: String,

    @ColumnInfo(name = "queued_at")
    val queuedAt: Long,

    @ColumnInfo(name = "refused_reason")
    val refusedReason: String? = null,
)

@Dao
interface SyncQueueDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun enqueue(item: SyncItemEntity)

    /** The oldest items of one kind still to send. */
    @Query(
        "SELECT * FROM sync_queue WHERE kind = :kind AND refused_reason IS NULL " +
            "ORDER BY queued_at, id LIMIT :limit",
    )
    suspend fun pending(kind: String, limit: Int): List<SyncItemEntity>

    @Query("DELETE FROM sync_queue WHERE id IN (:ids)")
    suspend fun remove(ids: List<String>)

    @Query("UPDATE sync_queue SET refused_reason = :reason WHERE id = :id")
    suspend fun refuse(id: String, reason: String)

    @Query("SELECT COUNT(*) FROM sync_queue WHERE refused_reason IS NULL")
    suspend fun pendingCount(): Int
}

/** What one push did. */
data class SyncOutcome(
    /** False when the server could not be reached or refused the push whole. */
    val reachedServer: Boolean,
    val sent: Int = 0,
    val accepted: Int = 0,
    /** Rejected for good ("invalid", "not_owned"): kept, not sent again. */
    val refused: Int = 0,
    /** Still queued for a later push. */
    val retained: Int = 0,
)

/**
 * The offline queue and its push.
 *
 * Report 09's contract: the server answers a push with one acknowledgement per
 * item, `{kind, index, id, status, reason}`, where index is the item's place in
 * its kind's list. An item leaves the queue only when its own acknowledgement,
 * matched on (kind, index), says accepted. A missing acknowledgement, a network
 * failure or a refused push changes nothing, so nothing is lost by trying.
 */
class SyncQueue(
    private val dao: SyncQueueDao,
    private val api: SyncApi,
    private val clock: () -> Long = System::currentTimeMillis,
) : SyncOutbox {

    private val pushing = Mutex()

    override suspend fun enqueue(item: SyncItem): String {
        val payload = item.payload + (item.kind.idField to item.id)
        dao.enqueue(
            SyncItemEntity(
                id = item.id,
                kind = item.kind.wire,
                payloadJson = payloadAdapter.toJson(payload),
                queuedAt = clock(),
            ),
        )
        return item.id
    }

    /**
     * Queue an observation of a patient, recorded while offline. Returns its id,
     * which the server deduplicates on, so pushing it twice records it once.
     */
    suspend fun enqueueObservation(
        patientId: String,
        category: String,
        entityName: String,
        valueText: String? = null,
        severity: Int? = null,
        timestampMillis: Long = clock(),
    ): String = enqueue(
        SyncItem(
            kind = SyncKind.OBSERVATION,
            id = UUID.randomUUID().toString(),
            payload = mapOf(
                "patient_id" to patientId,
                "category" to category,
                "entity_name" to entityName,
                "value_text" to valueText,
                "severity" to severity,
                "timestamp" to timestampMillis / 1000.0,
                "reported_by" to "app",
            ),
        ),
    )

    /** Items still waiting to be sent. */
    suspend fun pendingCount(): Int = dao.pendingCount()

    /** Push what is queued, and keep exactly what the server did not accept. */
    suspend fun push(): SyncOutcome = pushing.withLock {
        val batch = SyncKind.entries.associateWith { dao.pending(it.wire, MAX_PER_KIND) }
        val sent = batch.values.sumOf { it.size }
        if (sent == 0) return SyncOutcome(reachedServer = true)

        val body = SyncPushBody(
            observations = payloads(batch, SyncKind.OBSERVATION),
            medication_reminders = payloads(batch, SyncKind.MEDICATION_REMINDER),
            interaction_logs = payloads(batch, SyncKind.INTERACTION_LOG),
            vignette_responses = payloads(batch, SyncKind.VIGNETTE_RESPONSE),
            device_timestamp = clock() / 1000.0,
        )
        val response = try {
            api.push(body)
        } catch (error: IOException) {
            Log.w(TAG, "sync push failed; $sent items stay queued", error)
            return SyncOutcome(reachedServer = false, sent = sent, retained = sent)
        } catch (error: JsonDataException) {
            Log.w(TAG, "sync reply could not be read; $sent items stay queued", error)
            return SyncOutcome(reachedServer = false, sent = sent, retained = sent)
        }
        val result = response.body()
        if (!response.isSuccessful || result == null) {
            Log.w(TAG, "sync push refused (${response.code()}); $sent items stay queued")
            return SyncOutcome(reachedServer = false, sent = sent, retained = sent)
        }

        val accepted = mutableSetOf<String>()
        val refused = mutableMapOf<String, String>()
        for (ack in result.acknowledgements) {
            val item = SyncKind.fromWire(ack.kind)?.let { batch[it] }?.getOrNull(ack.index) ?: continue
            when {
                ack.status == ACCEPTED -> accepted.add(item.id)
                ack.reason in NEVER_ACCEPTED -> {
                    Log.w(TAG, "sync: ${ack.kind} ${ack.index} refused (${ack.reason}); kept, not resent")
                    refused[item.id] = ack.reason.orEmpty()
                }
            }
        }
        refused.keys.removeAll(accepted)
        if (accepted.isNotEmpty()) dao.remove(accepted.toList())
        for ((id, reason) in refused) dao.refuse(id, reason)
        SyncOutcome(
            reachedServer = true,
            sent = sent,
            accepted = accepted.size,
            refused = refused.size,
            retained = sent - accepted.size - refused.size,
        )
    }

    private fun payloads(batch: Map<SyncKind, List<SyncItemEntity>>, kind: SyncKind): List<Map<String, Any?>> =
        batch[kind].orEmpty().map { payloadAdapter.fromJson(it.payloadJson).orEmpty() }

    companion object {
        private const val TAG = "PalliSync"
        private const val ACCEPTED = "accepted"

        /** The server takes at most this many items of a kind per push. */
        const val MAX_PER_KIND = 50

        /** Rejections that no later push can turn into an acceptance. */
        private val NEVER_ACCEPTED = setOf("invalid", "not_owned")

        private val payloadAdapter = Moshi.Builder().build().adapter<Map<String, Any?>>(
            Types.newParameterizedType(Map::class.java, String::class.java, Any::class.java),
        )
    }
}

/**
 * An interaction the server never saw, as the interaction log the server's sync
 * stores (`evaluation/interaction_logger.py`). It carries what the study counts
 * and none of the free text: the question and answer stay on the phone.
 */
fun offlineInteractionLog(row: InteractionEntity): SyncItem = SyncItem(
    kind = SyncKind.INTERACTION_LOG,
    id = row.id,
    payload = mapOf(
        "session_id" to "",
        "event_type" to "offline_interaction",
        "event_data" to mapOf(
            "release_id" to row.releaseId,
            "release_approved" to row.releaseApproved,
            "care_role" to row.careRole,
            "channel" to row.channel,
            "from_cache" to row.fromCache,
            "answer_kind" to row.answerKind,
            "emergency_level" to row.emergencyLevel,
            "dosage_blocked" to row.dosageBlocked,
            "source_count" to row.sourceCount,
            "interaction_class" to row.interactionClass,
        ),
        "timestamp" to row.occurredAt / 1000.0,
        "language" to row.language,
        "site_id" to row.siteId,
        "is_offline" to true,
    ),
)
