package org.inventcures.pallisahayak.data.offline

import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.runTest
import org.inventcures.pallisahayak.data.local.InteractionEntity
import org.inventcures.pallisahayak.data.local.PalliDatabase
import org.inventcures.pallisahayak.safety.AnswerKind
import org.inventcures.pallisahayak.safety.EmergencySeverity
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Past interactions, readable with no connection.
 *
 * The property that matters is that a refusal stays a refusal. Someone rereading
 * history weeks later must not find a Redacted Answer rendered the same way as
 * clinical advice, and that is a rendering concern carried as data.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class HistoryRepositoryTest {

    private lateinit var db: PalliDatabase
    private lateinit var history: HistoryRepository

    @Before
    fun setUp() {
        PalliDatabase.resetForTests()
        db = Room.inMemoryDatabaseBuilder(
            ApplicationProvider.getApplicationContext(),
            PalliDatabase::class.java,
        )
            .setQueryExecutor { it.run() }
            .setTransactionExecutor { it.run() }
            .allowMainThreadQueries()
            .build()
        history = HistoryRepository(db.interactions())
    }

    @After
    fun tearDown() {
        db.close()
        PalliDatabase.resetForTests()
    }

    private fun row(
        id: String,
        answer: String = "Keep the room cool.",
        kind: String = "ANSWER",
        offline: Boolean = false,
        fromCache: Boolean = false,
        emergency: String = "none",
        at: Long = 1_700_000_000L,
    ) = InteractionEntity(
        id = id,
        participantId = "p-1",
        careRole = "asha_worker",
        siteId = "site-1",
        releaseId = "rel-1",
        releaseApproved = true,
        occurredAt = at,
        language = "en-IN",
        channel = "text",
        isOffline = offline,
        fromCache = fromCache,
        voicePath = "text",
        answerKind = kind,
        emergencyLevel = emergency,
        dosageBlocked = kind != "ANSWER",
        sourceCount = 1,
        interactionClass = "substantive",
        textIsScrubbed = true,
        query = "how is she?",
        response = answer,
    )

    @Test
    fun `history is readable with no network`() = runTest {
        db.interactions().insert(row("a"))
        db.interactions().insert(row("b", at = 1_700_000_100L))

        val entries = history.recent()

        assertEquals(2, entries.size)
    }

    @Test
    fun `a redacted answer from weeks ago is still a redacted answer`() = runTest {
        db.interactions().insert(
            row(
                "old-refusal",
                answer = "Ask your clinician about dosing.",
                kind = "REDACTED_ANSWER",
                at = 1_600_000_000L,
            ),
        )

        val entry = history.recent().single()

        // Criterion 2. The kind travelled with the row rather than being re-derived,
        // because re-running the guard now would judge a copy of the text, not the
        // answer as it was given, and could disagree with what was said then.
        assertEquals(AnswerKind.REDACTED_ANSWER, entry.kind)
    }

    @Test
    fun `a deferral stays a deferral`() = runTest {
        db.interactions().insert(row("d", answer = "I understand...", kind = "DEFERRAL"))

        assertEquals(AnswerKind.DEFERRAL, history.recent().single().kind)
    }

    @Test
    fun `an ordinary answer is not mistaken for a refusal`() = runTest {
        db.interactions().insert(row("plain", kind = "ANSWER"))

        assertEquals(AnswerKind.ANSWER, history.recent().single().kind)
    }

    @Test
    fun `an unknown kind from an older build reads as an answer`() = runTest {
        db.interactions().insert(row("legacy", kind = "SOME_KIND_FROM_THE_FUTURE"))

        // The fallback is deliberately the answer side. A kind this build does not
        // know was written by a build that has not shipped, so treating it as a
        // refusal would mislabel ordinary guidance. Answer styling for a refusal is
        // the worse of the two errors.
        assertEquals(AnswerKind.ANSWER, history.recent().single().kind)
    }

    @Test
    fun `offline and cached provenance survive into the entry`() = runTest {
        db.interactions().insert(row("off", offline = true, fromCache = true))

        val entry = history.recent().single()

        // Criterion 8: a cached answer is marked as such and never presented as live.
        assertTrue(entry.wasOffline)
        assertTrue(entry.cameFromCache)
    }

    @Test
    fun `emergency severity survives into the entry`() = runTest {
        db.interactions().insert(row("e", emergency = "critical"))

        assertEquals(
            EmergencySeverity.CRITICAL,
            history.recent().single().emergency,
        )
    }

    @Test
    fun `newest interactions come first`() = runTest {
        db.interactions().insert(row("old", at = 1_600_000_000L))
        db.interactions().insert(row("new", at = 1_700_000_000L))

        assertEquals(listOf("new", "old"), history.recent().map { it.id })
    }

    @Test
    fun `an empty history is reported as empty rather than failing`() = runTest {
        assertTrue(history.recent().isEmpty())
        assertFalse(history.hasHistory().first())
    }
}