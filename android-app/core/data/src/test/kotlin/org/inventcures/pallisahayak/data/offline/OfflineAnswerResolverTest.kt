package org.inventcures.pallisahayak.data.offline

import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import java.io.File
import java.security.MessageDigest
import kotlinx.coroutines.test.runTest
import org.inventcures.pallisahayak.data.local.CachedAnswerEntity
import org.inventcures.pallisahayak.data.local.PalliDatabase
import org.inventcures.pallisahayak.data.model.Language
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * The offline bundle, against a real database.
 *
 * The property that matters is honesty. An offline app that guesses is worse than
 * one that admits it cannot help, because a plausible wrong answer reads as clinical
 * advice. Most of what is here is about what happens when the bundle cannot answer.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class OfflineAnswerResolverTest {

    private lateinit var db: PalliDatabase
    private lateinit var resolver: OfflineAnswerResolver

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
        resolver = OfflineAnswerResolver(db.cachedAnswers())
    }

    @After
    fun tearDown() {
        db.close()
        PalliDatabase.resetForTests()
    }

    private fun row(
        text: String,
        language: Language,
        answer: String = "Keep the room cool and the patient comfortable.",
        kind: String = "ANSWER",
    ) = CachedAnswerEntity(
        queryHash = resolver.hashOf(text),
        language = language.tag,
        queryText = text,
        response = answer,
        answerKind = kind,
        bundleVersion = "v1",
        fetchedAt = 1_700_000_000L,
    )

    // -- the hash, which is the seam that was missing ------------------------

    @Test
    fun `the hash matches what the server builder produces`() {
        val expected = MessageDigest.getInstance("SHA-256")
            .digest("what is the emergency number".toByteArray())
            .joinToString("") { "%02x".format(it) }

        assertEquals("SHA-256 hex must be 64 characters", 64, expected.length)
        assertEquals(expected, resolver.hashOf("  What Is The Emergency Number?  "))
    }

    @Test
    fun `the builder still hashes the way this client hashes`() {
        // Read the builder rather than trusting this comment. The two halves live in
        // different languages and cannot share code, so a change on either side would
        // otherwise turn every offline lookup into a silent miss.
        val builder = File("../../../offline/cache_builder.py")
        assertTrue("cache_builder.py not found from ${File(".").absolutePath}", builder.exists())
        val text = builder.readText()
        assertTrue("builder must normalise before hashing", text.contains("_normalise_for_hash"))
        // The client strips a trailing question mark, so the builder must too, or
        // "How do I manage pain?" misses a bundle built without one.
        assertTrue(
            "builder must strip trailing question marks",
            text.contains("(\"?\", \"!\")"),
        )
    }

    @Test
    fun `case and surrounding space do not change the lookup`() {
        assertEquals(
            resolver.hashOf("how to manage pain at home"),
            resolver.hashOf("  How To Manage Pain At Home  "),
        )
    }

    @Test
    fun `different questions hash differently`() {
        assertNotEquals(
            resolver.hashOf("how to manage pain at home"),
            resolver.hashOf("how to manage nausea"),
        )
    }

    // -- resolving -----------------------------------------------------------

    @Test
    fun `a bundled question is answered from the bundle`() = runTest {
        db.cachedAnswers().insertAll(listOf(row("how to manage breathlessness", Language.ENGLISH)))

        val result = resolver.resolve("How to manage breathlessness?", Language.ENGLISH)

        assertTrue(result is OfflineResolution.Found)
        assertEquals("ANSWER", (result as OfflineResolution.Found).answer.answerKind)
    }

    @Test
    fun `a naturally typed question still matches the bundle`() = runTest {
        val canonical = "how to manage pain at home"
        db.cachedAnswers().insertAll(listOf(row(canonical, Language.HINDI)))

        // The bundle holds twenty canonical questions, not paraphrases of them.
        // What a person actually changes is capitalisation and a trailing question
        // mark, and without normalisation those miss every row, so the app tells a
        // worker to find a connection for a question it already answers.
        val result = resolver.resolve("How to manage pain at home?", Language.HINDI)

        assertTrue("typed phrasing should still match: $result", result is OfflineResolution.Found)
    }

    @Test
    fun `a question is not answered from another language's bundle`() = runTest {
        db.cachedAnswers().insertAll(listOf(row("how to manage breathlessness", Language.ENGLISH)))

        // Answering from the English bundle while the screen says Hindi would be a
        // silent substitution: the worker asked in one language and got another.
        val result = resolver.resolve("how to manage breathlessness", Language.HINDI)

        assertTrue(result is OfflineResolution.Missed)
    }

    @Test
    fun `a paraphrase is a miss rather than a guess`() = runTest {
        db.cachedAnswers().insertAll(listOf(row("how to manage pain at home", Language.HINDI)))

        // Stated so it is a decision rather than an oversight. Matching meaning would
        // need on-device retrieval, which ADR 0005 rules out for offline. So a
        // paraphrase misses and says why, which is honest, instead of returning the
        // nearest stored answer and implying it is what was asked.
        val result = resolver.resolve("what should I do about the pain?", Language.HINDI)

        assertTrue(result is OfflineResolution.Missed)
    }

    @Test
    fun `a question outside the bundle says so and asks for a connection`() = runTest {
        db.cachedAnswers().insertAll(listOf(row("how to manage pain at home", Language.HINDI)))

        val result = resolver.resolve("can I give ibuprofen?", Language.HINDI)

        assertTrue(result is OfflineResolution.Missed)
        assertEquals(OfflineMiss.NOT_IN_BUNDLE, (result as OfflineResolution.Missed).reason)
        assertEquals("can I give ibuprofen?", result.question)
    }

    @Test
    fun `no bundle at all is distinguished from a genuine miss`() = runTest {
        // Different remedies: one says sync, the other says find signal. Collapsing
        // them makes both useless.
        val result = resolver.resolve("how to manage pain at home", Language.TAMIL)

        assertEquals(
            OfflineMiss.NO_BUNDLE_DOWNLOADED,
            (result as OfflineResolution.Missed).reason,
        )
    }

    // -- cached answers keep their kind --------------------------------------

    @Test
    fun `a cached refusal keeps its answer kind`() = runTest {
        db.cachedAnswers().insertAll(
            listOf(
                row(
                    "what dose of morphine",
                    Language.ENGLISH,
                    answer = "Ask your clinician about dosing.",
                    kind = "REDACTED_ANSWER",
                ),
            ),
        )

        val found = resolver.resolve("what dose of morphine", Language.ENGLISH)
            as OfflineResolution.Found

        // This is what stops a refusal saved three weeks ago being rendered later as
        // clinical advice. The kind travels with the cached row rather than being
        // re-derived, because re-deriving means re-running a guard on an answer that
        // has already been judged.
        assertEquals("REDACTED_ANSWER", found.answer.answerKind)
    }

    @Test
    fun `bundle size and presence are reported for the offline banner`() = runTest {
        assertEquals(0, resolver.bundleSize(Language.BENGALI))
        assertTrue(!resolver.hasBundle(Language.BENGALI))

        db.cachedAnswers().insertAll(
            listOf(
                row("how to manage pain at home", Language.BENGALI),
                row("how to manage nausea", Language.BENGALI),
            ),
        )

        assertEquals(2, resolver.bundleSize(Language.BENGALI))
        assertTrue(resolver.hasBundle(Language.BENGALI))
    }
}