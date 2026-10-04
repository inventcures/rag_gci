package org.inventcures.pallisahayak.data.model

/**
 * The eleven languages this build supports, and nothing else.
 *
 * ADR 0002 bounds the set by what the speech service can actually speak. Adding a
 * twelfth means a voice the study cannot verify, so the list is closed and the
 * refusal is a real outcome rather than a gap to paper over.
 *
 * The order matches `SUPPORTED_LANGUAGES` in `offline/questions.py` exactly. That
 * list is zipped against the offline question lists to build the cache bundle, so a
 * mismatch in length or order would pair the wrong question with the wrong answer
 * and nothing would raise. `LanguageCatalogTest` asserts the two agree by name.
 */
enum class Language(
    /** BCP-47 tag, as the server, the index and the speech service all use it. */
    val tag: String,
    /** Endonym. A worker finds their own language faster than a translated name. */
    val endonym: String,
) {
    ENGLISH("en-IN", "English"),
    HINDI("hi-IN", "हिन्दी"),
    BENGALI("bn-IN", "বাংলা"),
    KANNADA("kn-IN", "ಕನ್ನಡ"),
    MALAYALAM("ml-IN", "മലയാളം"),
    MARATHI("mr-IN", "मराठी"),
    ODIA("od-IN", "ଓଡ଼ିଆ"),
    PUNJABI("pa-IN", "ਪੰਜਾਬੀ"),
    TAMIL("ta-IN", "தமிழ்"),
    TELUGU("te-IN", "తెలుగు"),
    GUJARATI("gu-IN", "ગુજરાતી");

    companion object {
        val ALL: List<Language> = entries

        /**
         * Resolve a tag, or null when it is not supported.
         *
         * Null rather than a default on purpose. Criterion: an unsupported language
         * is refused plainly and never silently substituted. Returning English here
         * would answer a Marathi worker in English while the screen said Marathi,
         * which is the worst outcome available: they believe they were understood.
         *
         * Matching is case-insensitive and tolerates a bare region ("hi" as well as
         * "hi-IN"), because a locale can arrive from the device rather than from us.
         */
        fun fromTag(tag: String?): Language? {
            val normalised = tag?.trim()?.lowercase() ?: return null
            return ALL.firstOrNull { it.tag.lowercase() == normalised }
                ?: ALL.firstOrNull { it.tag.substringBefore('-').lowercase() == normalised }
        }

        fun isSupported(tag: String?): Boolean = fromTag(tag) != null
    }
}