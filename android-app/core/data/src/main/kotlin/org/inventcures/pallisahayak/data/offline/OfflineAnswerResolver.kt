package org.inventcures.pallisahayak.data.offline

import org.inventcures.pallisahayak.data.local.CachedAnswerDao
import org.inventcures.pallisahayak.data.local.CachedAnswerEntity
import org.inventcures.pallisahayak.data.model.Language
import java.security.MessageDigest

/** Why a question could not be answered offline. */
enum class OfflineMiss {
    /** Nothing downloaded yet for this language. Different from a genuine miss. */
    NO_BUNDLE_DOWNLOADED,

    /** The bundle is there but does not cover this question. */
    NOT_IN_BUNDLE,
}

/** What the bundle can do about a question, decided without the network. */
sealed interface OfflineResolution {
    data class Found(val answer: CachedAnswerEntity) : OfflineResolution
    data class Missed(val reason: OfflineMiss, val question: String, val language: Language) :
        OfflineResolution
}

/**
 * Resolves a question against the downloaded bundle.
 *
 * ADR 0005: offline means a downloaded bundle of anticipated answers plus history,
 * not on-device retrieval. The app is then honest about the edge of it. A question
 * the bundle does not cover gets an explanation that it needs a connection, because
 * silence reads as a freeze and a wrong answer reads as clinical advice.
 */
class OfflineAnswerResolver(private val dao: CachedAnswerDao) {

    /**
     * The one place a question becomes a lookup key.
     *
     * `offline/cache_builder.py:99` hashes the lowercased, stripped question with
     * SHA-256. Nothing on the Android side computed that, so every lookup could
     * miss and an offline app would report every question as absent, which looks
     * exactly like an empty bundle. Duplicated rather than shared because the
     * client cannot import server code; pinned by a test that reads the builder.
     */
    fun hashOf(question: String): String {
        val normalised = normalise(question)
        val digest = MessageDigest.getInstance("SHA-256").digest(normalised.toByteArray())
        return digest.joinToString("") { "%02x".format(it) }
    }

    /**
     * Reduce a typed question to what the bundle was built from.
     *
     * Lowercasing and trimming are what the builder does. Trailing punctuation is
     * stripped as well, because people type "How do I manage pain?" and the bundle
     * holds "how to manage pain at home" with no question mark. Without this a
     * perfectly ordinary question misses the bundle and the app says it needs a
     * connection, which is both wrong and unfixable from the user's side.
     *
     * The builder applies the same normalisation. The two cannot share code, so the
     * test below reads the builder and fails if the two drift.
     */
    private fun normalise(question: String): String =
        question
            .lowercase()
            .trim()
            .removeSuffix("?")
            .removeSuffix("!")
            .trim()

    suspend fun resolve(question: String, language: Language): OfflineResolution {
        val hit = dao.find(hashOf(question), language.tag)
        if (hit != null) return OfflineResolution.Found(hit)

        // "No questions downloaded" and "not one of them" need different remedies,
        // so they must not collapse into one message.
        val downloaded = dao.countForLanguage(language.tag) > 0
        return OfflineResolution.Missed(
            reason = if (downloaded) OfflineMiss.NOT_IN_BUNDLE else OfflineMiss.NO_BUNDLE_DOWNLOADED,
            question = question,
            language = language,
        )
    }

    suspend fun bundleSize(language: Language): Int = dao.countForLanguage(language.tag)

    suspend fun hasBundle(language: Language): Boolean = dao.countForLanguage(language.tag) > 0
}
