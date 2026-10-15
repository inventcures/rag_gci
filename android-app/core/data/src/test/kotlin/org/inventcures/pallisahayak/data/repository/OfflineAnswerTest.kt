package org.inventcures.pallisahayak.data.repository

import kotlinx.coroutines.test.runTest
import org.inventcures.pallisahayak.api.generated.CacheBundleResponse
import org.inventcures.pallisahayak.data.local.CachedAnswerEntity
import org.inventcures.pallisahayak.data.model.Language
import org.inventcures.pallisahayak.data.offline.OfflineAnswerResolver
import org.inventcures.pallisahayak.data.offline.OfflineBundleDownloader
import org.inventcures.pallisahayak.safety.AnswerKind
import org.inventcures.pallisahayak.safety.EmergencySeverity
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * With no signal, the app answers from the Cached Answer Bundle (ADR 0005), and
 * says so.
 *
 * Before ticket 23 the repository never looked at the bundle: every offline
 * question became "no connection", and nothing downloaded a bundle either.
 */
class OfflineAnswerTest {

    private val api = FakeApi()
    private val interactions = FakeInteractionDao()
    private val cached = FakeCachedAnswerDao()
    private val resolver = OfflineAnswerResolver(cached)
    private val repository = PalliSahayakRepository(
        api = api,
        interactions = interactions,
        bundle = resolver,
    )

    private val session = SessionContext(
        participantId = "3f2a8c14-9b7e-4d21-8a55-6c1e0d3b7f92",
        careRole = "family_caregiver",
        siteId = "CCHRC",
        language = "hi-IN",
        releaseId = "study-1@abc1234",
        releaseApproved = true,
    )

    private val bundledQuestion = "कब्ज़ में क्या करें"
    private val bundledAnswer =
        "पानी और रेशेदार भोजन बढ़ाएँ, और रोज़ थोड़ा चलने में मदद करें। डॉक्टर से अपनी दवाओं के बारे में पूछें।"

    private fun bundle(answer: String = bundledAnswer, kind: String = "ANSWER") {
        cached.rows.add(
            CachedAnswerEntity(
                queryHash = resolver.hashOf(bundledQuestion),
                language = "hi-IN",
                queryText = bundledQuestion,
                response = answer,
                answerKind = kind,
                bundleVersion = "v1",
                fetchedAt = 1_700_000_000L,
            ),
        )
    }

    @Test
    fun `in airplane mode a bundled question is answered from the bundle and marked as such`() = runTest {
        // A plain answer with no schedule words, so it passes the boundary whole.
        val plain = "Keep the room cool, offer sips of water, and sit with the patient."
        bundle(answer = plain)
        api.offline = true

        val answered = repository.ask("$bundledQuestion?", session)

        assertTrue(answered.source is AnswerSource.Bundle)
        assertEquals(plain, answered.text)
        assertEquals(AnswerKind.ANSWER, answered.kind)
        val row = interactions.rows.single()
        assertTrue("recorded as from the bundle", row.fromCache)
        assertTrue("recorded as offline", row.isOffline)
        assertEquals("substantive", row.interactionClass)
    }

    @Test
    fun `a bundled answer still passes the Dose Boundary`() = runTest {
        bundle(answer = "मॉर्फिन 10 मिलीग्राम दें।")
        api.offline = true

        val answered = repository.ask(bundledQuestion, session)

        assertTrue(answered.source is AnswerSource.Bundle)
        assertNotEquals(AnswerKind.ANSWER, answered.kind)
        assertFalse(answered.text.contains("10 मिलीग्राम"))
    }

    @Test
    fun `a bundled deferral is shown as a deferral, never as an answer`() = runTest {
        bundle(kind = "DEFERRAL")
        api.offline = true

        val answered = repository.ask(bundledQuestion, session)

        assertEquals(AnswerKind.DEFERRAL, answered.kind)
    }

    @Test
    fun `offline, a question outside the bundle is unavailable rather than guessed`() = runTest {
        bundle()
        api.offline = true

        val answered = repository.ask("कुछ और पूछना है", session)

        assertEquals(AnswerSource.Unavailable, answered.source)
        assertEquals(AnswerKind.DEFERRAL, answered.kind)
    }

    @Test
    fun `offline, an emergency question keeps its CRITICAL level`() = runTest {
        api.offline = true

        val answered = repository.ask("मरीज़ बेहोश है", session)

        assertEquals(EmergencySeverity.CRITICAL, answered.emergency)
    }

    @Test
    fun `a bundle is downloaded for the participant's language and replaces the older one`() = runTest {
        cached.rows.add(
            CachedAnswerEntity("old", "hi-IN", "old question", "old answer", "ANSWER", "v0", 1L),
        )
        cached.rows.add(
            CachedAnswerEntity("other", "ta-IN", "tamil question", "tamil answer", "ANSWER", "t1", 1L),
        )
        api.bundle = CacheBundleResponse(
            emergency_keywords = emptyMap(),
            evidence_badge_metadata = emptyMap(),
            generated_at = 1_700_000_000.0,
            language = "hi",
            queries = listOf(
                mapOf(
                    "query_hash" to resolver.hashOf(bundledQuestion),
                    "query_text" to bundledQuestion,
                    "response_text" to bundledAnswer,
                ),
                mapOf(
                    "query_hash" to "dosed",
                    "query_text" to "dose question",
                    "response_text" to "Give 10 mg morphine now.",
                ),
                // No answer text: not stored, rather than stored empty.
                mapOf("query_hash" to "empty", "query_text" to "empty"),
            ),
            treatments = emptyList(),
            version = "v2",
        )

        val stored = OfflineBundleDownloader(api, cached).refresh(Language.HINDI)

        assertEquals(listOf("hi-IN"), api.bundleRequests)
        assertEquals(2, stored)
        assertEquals(setOf(resolver.hashOf(bundledQuestion), "dosed", "other"), cached.rows.map { it.queryHash }.toSet())
        assertEquals("DEFERRAL", cached.rows.single { it.queryHash == "dosed" }.answerKind)
        assertEquals("v2", cached.rows.single { it.queryHash == "dosed" }.bundleVersion)
    }

    @Test
    fun `a failed download leaves the previous bundle usable`() = runTest {
        bundle()
        api.offline = true

        val stored = OfflineBundleDownloader(api, cached).refresh(Language.HINDI)

        assertEquals(null, stored)
        assertEquals(1, cached.rows.size)
    }
}
