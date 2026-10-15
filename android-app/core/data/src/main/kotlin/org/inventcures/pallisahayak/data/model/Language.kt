package org.inventcures.pallisahayak.data.model

/**
 * The eleven languages this build supports, and nothing else.
 *
 * ADR 0002 bounds the set by what the speech service can actually speak. Adding a
 * twelfth means a voice the study cannot verify, so the list is closed and the
 * refusal is a real outcome rather than a gap to paper over.
 *
 * The tags are the Supported Language table's (`supported_languages.py`), so Odia
 * is "or-IN", not Sarvam's "od-IN": the server's bundle, index and logs key Odia
 * by the table's tag, and the app's own Odia string resources live under the
 * table's code (`values-or`). `LanguageAndRoleTest` asserts the two sets agree.
 * The order is the picker's; the offline bundle is keyed by tag, not by position.
 */
enum class Language(
    /** BCP-47 tag, as the Supported Language table writes it, e.g. `or-IN`. */
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
    ODIA("or-IN", "ଓଡ଼ିଆ"),
    PUNJABI("pa-IN", "ਪੰਜਾਬੀ"),
    TAMIL("ta-IN", "தமிழ்"),
    TELUGU("te-IN", "తెలుగు"),
    GUJARATI("gu-IN", "ગુજરાતી");

    /**
     * The table's short code, e.g. `or`. It is also the Android resource qualifier
     * and the locale the app's screens are shown in for this language.
     */
    val code: String get() = tag.substringBefore('-')

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
            val trimmed = tag?.trim()?.lowercase()?.replace('_', '-') ?: return null
            val normalised = ALIASES[trimmed] ?: trimmed
            return ALL.firstOrNull { it.tag.lowercase() == normalised }
                ?: ALL.firstOrNull { it.code == normalised }
        }

        /**
         * Other spellings of a Supported Language, as the table's `aliases`.
         *
         * Builds before ticket 24 stored and sent Odia as "od-IN" (Sarvam's code), so
         * a phone upgraded from one still holds it; it is read as Odia, never refused.
         */
        private val ALIASES: Map<String, String> = mapOf("od" to "or-in", "od-in" to "or-in")

        fun isSupported(tag: String?): Boolean = fromTag(tag) != null
    }
}