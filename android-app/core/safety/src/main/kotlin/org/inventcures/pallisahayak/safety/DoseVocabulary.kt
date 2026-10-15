package org.inventcures.pallisahayak.safety

/**
 * The dose vocabulary, word for word the server's (`dose_detector.py`).
 *
 * Plain words, one script per literal, so a mixed-script or corrupted entry is
 * visible in review. `DoseVocabularyTest` reads the server's source and fails on
 * any difference, because the client cannot import server code and a copy that
 * drifts is exactly how the Android detector came to miss what the server caught.
 * Change a word here only together with the same change in `dose_detector.py`.
 */
object DoseVocabulary {

    /** English number words, so "five milligrams" is a dose like "5 mg". */
    val EN_NUMBER_WORDS: List<String> = listOf(
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
        "hundred", "thousand", "half", "quarter", "dozen",
    )

    /** Hindi number words, matched only as whole words. */
    val HI_NUMBER_WORDS: List<String> = listOf(
        "एक", "दो", "तीन", "चार", "पाँच", "पांच", "छह", "छः", "सात", "आठ", "नौ", "दस", "ग्यारह",
        "बारह", "पंद्रह", "बीस", "पच्चीस", "तीस", "चालीस", "पचास", "सौ", "आधा", "आधी", "डेढ़",
        "ढाई",
    )

    /**
     * The unit words of every Supported Language, in that language's own script.
     * Keys are the server's: Odia is `or`. Thai is not a Supported Language; its
     * units stay because dropping a pattern never makes the boundary safer.
     */
    val UNIT_WORDS: Map<String, List<String>> = mapOf(
        "hi" to listOf(
            "माइक्रोग्राम", "मिलीग्राम", "मिलिग्राम", "मिग्रा", "मि.ग्रा.", "ग्राम", "मिलीलीटर",
            "मिलिलीटर", "मि.ली.", "मिली", "एमजी", "एम.जी.", "एम जी", "एमएल", "एम.एल.", "एमसीजी",
        ),
        "mr" to listOf(
            "मायक्रोग्रॅम", "मिलिग्रॅम", "मि.ग्रॅ.", "ग्रॅम", "मिलिलिटर", "मि.लि.",
        ),
        "bn" to listOf(
            "মাইক্রোগ্রাম", "মিলিগ্রাম", "মি.গ্রা.", "গ্রাম", "মিলিলিটার", "মি.লি.", "মিলি",
            "এমজি", "এম.জি.",
        ),
        "pa" to listOf(
            "ਮਾਈਕ੍ਰੋਗ੍ਰਾਮ", "ਮਿਲੀਗ੍ਰਾਮ", "ਮਿ.ਗ੍ਰਾ.", "ਮਿਗ੍ਰਾ", "ਗ੍ਰਾਮ", "ਮਿਲੀਲੀਟਰ", "ਮਿ.ਲੀ.",
            "ਮਿਲੀ", "ਐਮਜੀ", "ਐਮ.ਜੀ.",
        ),
        "gu" to listOf(
            "માઇક્રોગ્રામ", "મિલિગ્રામ", "મિલીગ્રામ", "મિ.ગ્રા.", "ગ્રામ", "મિલિલિટર", "મિ.લી.",
            "મિલી", "એમજી", "એમ.જી.",
        ),
        "or" to listOf(
            "ମାଇକ୍ରୋଗ୍ରାମ", "ମିଲିଗ୍ରାମ", "ମି.ଗ୍ରା.", "ମିଗ୍ରା", "ଗ୍ରାମ", "ମିଲିଲିଟର", "ମି.ଲି.",
            "ମିଲି",
        ),
        "ta" to listOf(
            "மைக்ரோகிராம்", "மில்லிகிராம்", "மி.கி.", "கிராம்", "மில்லிலிட்டர்", "மி.லி.",
            "மில்லி",
        ),
        "te" to listOf(
            "మైక్రోగ్రాము", "మిల్లీగ్రాము", "మిల్లీగ్రామ్", "మి.గ్రా.", "గ్రాము", "గ్రామ్",
            "మిల్లీలీటర్", "మి.లీ.", "మిల్లీ",
        ),
        "kn" to listOf(
            "ಮೈಕ್ರೋಗ್ರಾಂ", "ಮಿಲಿಗ್ರಾಂ", "ಮಿಲಿಗ್ರಾಮ್", "ಮಿ.ಗ್ರಾ.", "ಗ್ರಾಂ", "ಗ್ರಾಮ್", "ಮಿಲಿಲೀಟರ್",
            "ಮಿ.ಲೀ.", "ಮಿಲಿ",
        ),
        "ml" to listOf(
            "മൈക്രോഗ്രാം", "മില്ലിഗ്രാം", "മി.ഗ്രാ.", "ഗ്രാം", "മില്ലിലിറ്റർ", "മി.ലി.", "മില്ലി",
        ),
        "th" to listOf(
            "มิลลิกรัม", "มิลลิลิตร",
        ),
    )

    /** Hindi counts of a dose form, with their plural and oblique forms. */
    val HI_COUNT_NOUNS: List<String> = listOf(
        "गोलियाँ", "गोलियां", "गोली", "टैबलेट", "टेबलेट", "कैप्सूल", "चम्मच", "बूँदें", "बूंदें",
        "बूँद", "बूंद", "पफ", "खुराक", "गोलियों", "बूँदों", "बूंदों", "खुराकें", "खुराकों",
    )

    /**
     * Day and time-of-day words that mark a schedule in Indic answers. Matched
     * whole, so each inflected form a schedule uses is listed in its own right.
     */
    val INDIC_FREQUENCY_WORDS: Map<String, List<String>> = mapOf(
        "hi" to listOf(
            "दिनों", "दिन", "रोज़", "रोज", "सुबह", "शाम", "रात", "बार", "रोज़ाना", "रोजाना",
            "प्रतिदिन", "दिनभर", "रात्रि", "रातों",
        ),
        "mr" to listOf(
            "रात्री", "रात्रीच्या", "दररोज",
        ),
        "kn" to listOf(
            "ದಿನ", "ದೀನ", "ಬಾರ", "ಬಾರಿ", "ದಿನಕ್ಕೆ", "ದಿನವೂ", "ದಿನಗಳು", "ಪ್ರತಿದಿನ", "ದಿನಾ",
        ),
        "ta" to listOf(
            "தினம்", "தின", "நாள்", "வாரம்", "முறை", "தினமும்", "தினசரி", "நாளுக்கு", "நாட்கள்",
            "முறைகள்", "வாரத்திற்கு", "வாரத்துக்கு",
        ),
        "te" to listOf(
            "రోజు", "రోజ", "తరువాత", "సార్లు", "రోజుకు", "రోజూ", "రోజులు", "ప్రతిరోజు",
            "ప్రతిరోజూ", "సారి",
        ),
        "pa" to listOf(
            "ਰੋਜ਼", "ਰੋਜ", "ਵਾਰ", "ਰੋਜ਼ਾਨਾ", "ਰੋਜਾਨਾ", "ਵਾਰੀ", "ਦਿਨ",
        ),
        "gu" to listOf(
            "રોજ", "દિવસ", "વાર", "દરરોજ", "રોજના", "દિવસમાં", "દિવસે", "દિવસો",
        ),
        "or" to listOf(
            "ଦିନ", "ରୋଜ", "ପ୍ରତିଦିନ", "ଦିନରେ", "ଦିନକୁ", "ଦିନେ",
        ),
    )

    /** Every Indic and English word above, for the vocabulary checks. */
    fun all(): List<String> =
        EN_NUMBER_WORDS + HI_NUMBER_WORDS + HI_COUNT_NOUNS +
            UNIT_WORDS.values.flatten() + INDIC_FREQUENCY_WORDS.values.flatten()
}
