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
     * Number words of the other Supported Languages, matched whole: one to ten,
     * the halves, and the forms a count takes before a noun ("দুটি", "ଦୁଇଟି").
     * Words shared with Hindi are listed once, in [HI_NUMBER_WORDS].
     */
    val INDIC_NUMBER_WORDS: Map<String, List<String>> = mapOf(
        "mr" to listOf(
            "दोन", "पाच", "सहा", "नऊ", "दहा", "अर्धा", "अर्धी", "अर्धे", "दीड", "अडीच", "पाव",
        ),
        "bn" to listOf(
            "এক", "একটি", "একটা", "দুই", "দু", "দুটি", "দুটো", "তিন", "তিনটি", "তিনটে", "চার",
            "চারটি", "চারটে", "পাঁচ", "পাঁচটি", "ছয়", "সাত", "আট", "নয়", "দশ", "আধা", "আধ",
            "অর্ধেক", "দেড়", "আড়াই",
        ),
        "pa" to listOf(
            "ਇੱਕ", "ਇਕ", "ਦੋ", "ਤਿੰਨ", "ਚਾਰ", "ਪੰਜ", "ਛੇ", "ਸੱਤ", "ਅੱਠ", "ਨੌਂ", "ਦਸ", "ਅੱਧੀ",
            "ਅੱਧਾ", "ਡੇਢ", "ਢਾਈ",
        ),
        "gu" to listOf(
            "એક", "બે", "ત્રણ", "ચાર", "પાંચ", "છ", "સાત", "આઠ", "નવ", "દસ", "અડધી", "અડધો",
            "અડધું", "અર્ધી", "અર્ધો", "દોઢ", "અઢી",
        ),
        "or" to listOf(
            "ଏକ", "ଗୋଟିଏ", "ଦୁଇ", "ଦୁଇଟି", "ଦୁଇଟା", "ତିନି", "ତିନିଟି", "ଚାରି", "ଚାରିଟି", "ପାଞ୍ଚ",
            "ଛଅ", "ସାତ", "ଆଠ", "ନଅ", "ଦଶ", "ଅଧା", "ଦେଢ଼", "ଅଢ଼େଇ",
        ),
        "ta" to listOf(
            "ஒன்று", "ஒரு", "இரண்டு", "இரு", "மூன்று", "மூணு", "நான்கு", "நாலு", "ஐந்து", "ஆறு",
            "ஏழு", "எட்டு", "ஒன்பது", "பத்து", "அரை", "பாதி", "கால்", "ஒன்றரை",
        ),
        "te" to listOf(
            "ఒకటి", "ఒక", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది",
            "పది", "అర", "సగం", "ఒకటిన్నర", "రెండున్నర",
        ),
        "kn" to listOf(
            "ಒಂದು", "ಎರಡು", "ಮೂರು", "ನಾಲ್ಕು", "ಐದು", "ಆರು", "ಏಳು", "ಎಂಟು", "ಒಂಬತ್ತು", "ಹತ್ತು",
            "ಅರ್ಧ", "ಒಂದೂವರೆ", "ಎರಡೂವರೆ",
        ),
        "ml" to listOf(
            "ഒന്ന്", "ഒരു", "രണ്ട്", "രണ്ടു", "മൂന്ന്", "മൂന്നു", "നാല്", "നാലു", "അഞ്ച്", "അഞ്ചു",
            "ആറ്", "ഏഴ്", "എട്ട്", "ഒമ്പത്", "പത്ത്", "അര", "പകുതി", "ഒന്നര",
        ),
    )

    /** The count words Bengali and Odia join to a digit ("২টি ট্যাবলেট"). */
    val INDIC_COUNT_CLASSIFIERS: List<String> = listOf(
        "টি", "টা", "টো", "টে", "ଟି", "ଟା",
    )

    /**
     * Counts of a dose form in the other Supported Languages (tablet, capsule,
     * drop, spoon, puff, dose) with their plural and case forms. They count only
     * after a number, so "the tablet helps" is not a dose.
     */
    val INDIC_COUNT_NOUNS: Map<String, List<String>> = mapOf(
        "mr" to listOf(
            "गोळी", "गोळ्या", "गोळ्यां", "गोळ्यांचा", "गोळ्यांची", "गोळीचा", "टॅबलेट", "टॅब्लेट",
            "कॅप्सूल", "थेंब", "चमचा", "चमचे", "डोस",
        ),
        "bn" to listOf(
            "ট্যাবলেট", "ট্যাবলেটের", "ট্যাবলেটে", "বড়ি", "বড়ির", "ক্যাপসুল", "ক্যাপসুলের",
            "ফোঁটা", "ফোঁটার", "চামচ", "চামচের", "পাফ", "ডোজ",
        ),
        "pa" to listOf(
            "ਗੋਲੀ", "ਗੋਲੀਆਂ", "ਟੈਬਲੇਟ", "ਕੈਪਸੂਲ", "ਬੂੰਦ", "ਬੂੰਦਾਂ", "ਤੁਪਕੇ", "ਚਮਚ", "ਚਮਚਾ", "ਚਮਚੇ",
            "ਪਫ", "ਖੁਰਾਕ", "ਖੁਰਾਕਾਂ",
        ),
        "gu" to listOf(
            "ગોળી", "ગોળીઓ", "ટેબ્લેટ", "ટેબલેટ", "કેપ્સ્યુલ", "ટીપાં", "ટીપું", "ચમચી", "ચમચા",
            "પફ", "ડોઝ",
        ),
        "or" to listOf(
            "ଟାବଲେଟ", "ଟାବଲେଟ୍", "ବଟିକା", "ଗୋଳି", "କ୍ୟାପସୁଲ", "ଟୋପା", "ଚାମଚ", "ପଫ୍", "ଡୋଜ୍",
        ),
        "ta" to listOf(
            "மாத்திரை", "மாத்திரைகள்", "மாத்திரைகளை", "மாத்திரையை", "மாத்திரைக்கு", "கேப்ஸ்யூல்",
            "கேப்சூல்", "சொட்டு", "சொட்டுகள்", "துளி", "துளிகள்", "ஸ்பூன்", "கரண்டி", "தேக்கரண்டி",
            "பஃப்", "டோஸ்",
        ),
        "te" to listOf(
            "మాత్ర", "మాత్రలు", "మాత్రను", "మాత్రలను", "టాబ్లెట్", "టాబ్లెట్లు", "ట్యాబ్లెట్",
            "ట్యాబ్లెట్లు", "క్యాప్సూల్", "చుక్క", "చుక్కలు", "చెంచా", "స్పూన్", "పఫ్", "డోస్",
        ),
        "kn" to listOf(
            "ಮಾತ್ರೆ", "ಮಾತ್ರೆಗಳು", "ಮಾತ್ರೆಗಳನ್ನು", "ಮಾತ್ರೆಯನ್ನು", "ಟ್ಯಾಬ್ಲೆಟ್", "ಕ್ಯಾಪ್ಸೂಲ್",
            "ಹನಿ", "ಹನಿಗಳು", "ಚಮಚ", "ಪಫ್", "ಡೋಸ್",
        ),
        "ml" to listOf(
            "ഗുളിക", "ഗുളികകൾ", "ഗുളികയും", "ടാബ്ലെറ്റ്", "ക്യാപ്സ്യൂൾ", "തുള്ളി", "തുള്ളികൾ",
            "സ്പൂൺ", "ടീസ്പൂൺ", "പഫ്", "ഡോസ്",
        ),
    )

    /**
     * Words that make "<number> <word>" a schedule: "N times" ("তিন বার") and
     * "once every N hours" as one word or a fixed phrase ("గంటలకోసారి").
     */
    val INDIC_TIMES_WORDS: Map<String, List<String>> = mapOf(
        "mr" to listOf(
            "वेळा",
        ),
        "bn" to listOf(
            "বার", "ঘণ্টা অন্তর", "ঘন্টা অন্তর", "ঘণ্টা পর পর", "ঘন্টা পর পর", "ঘণ্টা পরপর",
            "ঘন্টা পরপর",
        ),
        "pa" to listOf(
            "ਦਫ਼ਾ", "ਦਫਾ",
        ),
        "gu" to listOf(
            "વખત",
        ),
        "or" to listOf(
            "ଥର",
        ),
        "ta" to listOf(
            "தடவை",
        ),
        "te" to listOf(
            "మార్లు", "గంటలకోసారి", "గంటకోసారి", "గంటలకు ఒకసారి",
        ),
        "kn" to listOf(
            "ಸಲ", "ಸಾರಿ", "ಗಂಟೆಗೊಮ್ಮೆ", "ಗಂಟೆಗಳಿಗೊಮ್ಮೆ",
        ),
        "ml" to listOf(
            "തവണ", "പ്രാവശ്യം", "നേരം", "മണിക്കൂർ ഇടവിട്ട്",
        ),
    )

    /** "Every", before a number and an interval word: "दर चार तासांनी". */
    val INDIC_EVERY_WORDS: Map<String, List<String>> = mapOf(
        "hi" to listOf(
            "हर", "प्रति", "प्रत्येक",
        ),
        "mr" to listOf(
            "दर",
        ),
        "bn" to listOf(
            "প্রতি", "প্রত্যেক",
        ),
        "pa" to listOf(
            "ਹਰ", "ਹਰੇਕ", "ਪ੍ਰਤੀ",
        ),
        "gu" to listOf(
            "દર", "દરેક", "પ્રત્યેક",
        ),
        "or" to listOf(
            "ପ୍ରତି", "ପ୍ରତ୍ୟେକ",
        ),
        "ta" to listOf(
            "ஒவ்வொரு",
        ),
        "te" to listOf(
            "ప్రతి",
        ),
        "kn" to listOf(
            "ಪ್ರತಿ",
        ),
        "ml" to listOf(
            "ഓരോ",
        ),
    )

    /**
     * Hours, minutes and days after "every <number>". Without "every" they are a
     * time span ("rest for two hours"), not a schedule.
     */
    val INDIC_INTERVAL_WORDS: Map<String, List<String>> = mapOf(
        "hi" to listOf(
            "घंटे", "घंटों", "घंटा", "मिनट", "दिनों",
        ),
        "mr" to listOf(
            "तास", "तासांनी", "तासाने", "तासाला", "तासांनंतर", "मिनिटांनी", "दिवसांनी",
        ),
        "bn" to listOf(
            "ঘণ্টা", "ঘন্টা", "ঘণ্টায়", "ঘন্টায়", "ঘণ্টার", "মিনিট", "মিনিটে", "দিন", "দিনে",
        ),
        "pa" to listOf(
            "ਘੰਟੇ", "ਘੰਟਿਆਂ", "ਘੰਟਾ", "ਮਿੰਟ", "ਮਿੰਟਾਂ", "ਦਿਨਾਂ",
        ),
        "gu" to listOf(
            "કલાક", "કલાકે", "કલાકમાં", "કલાકના", "મિનિટે", "દિવસે",
        ),
        "or" to listOf(
            "ଘଣ୍ଟା", "ଘଣ୍ଟାରେ", "ଘଣ୍ଟାର", "ମିନିଟ", "ମିନିଟରେ",
        ),
        "ta" to listOf(
            "மணி", "மணிநேரம்", "மணிநேரத்திற்கும்", "மணிநேரத்திற்கு", "மணிநேரத்துக்கு", "மணிக்கு",
            "நிமிடம்", "நிமிடத்திற்கு", "நாட்களுக்கு",
        ),
        "te" to listOf(
            "గంటలకు", "గంటకు", "గంటల", "గంటలు", "గంట", "నిమిషాలకు", "రోజులకు",
        ),
        "kn" to listOf(
            "ಗಂಟೆಗೆ", "ಗಂಟೆಗಳಿಗೆ", "ಗಂಟೆ", "ಗಂಟೆಗಳು", "ನಿಮಿಷಕ್ಕೆ", "ದಿನಗಳಿಗೆ",
        ),
        "ml" to listOf(
            "മണിക്കൂറിലും", "മണിക്കൂറിൽ", "മണിക്കൂർ", "മണിക്കൂറിനും", "മിനിറ്റിലും", "ദിവസത്തിലും",
        ),
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
            "रात्री", "रात्रीच्या", "दररोज", "दोनदा", "तीनदा", "चारदा", "दिवसातून",
        ),
        "kn" to listOf(
            "ದಿನ", "ದೀನ", "ಬಾರ", "ಬಾರಿ", "ದಿನಕ್ಕೆ", "ದಿನವೂ", "ದಿನಗಳು", "ಪ್ರತಿದಿನ", "ದಿನಾ",
        ),
        "ta" to listOf(
            "தினம்", "தின", "நாள்", "வாரம்", "முறை", "தினமும்", "தினசரி", "நாளுக்கு", "நாட்கள்",
            "முறைகள்", "வாரத்திற்கு", "வாரத்துக்கு", "இரண்டு வேளை", "மூன்று வேளை", "மூணு வேளை",
            "நான்கு வேளை",
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
        "bn" to listOf(
            "দুবার", "দুইবার", "তিনবার", "চারবার", "দিনে একবার", "প্রতিদিন", "রোজ",
        ),
        "ml" to listOf(
            "ദിവസവും", "ദിവസേന", "പ്രതിദിനം",
        ),
    )

    /** Every Indic and English word above, for the vocabulary checks. */
    fun all(): List<String> =
        EN_NUMBER_WORDS + HI_NUMBER_WORDS + HI_COUNT_NOUNS + INDIC_COUNT_CLASSIFIERS +
            UNIT_WORDS.values.flatten() + INDIC_FREQUENCY_WORDS.values.flatten() +
            INDIC_NUMBER_WORDS.values.flatten() + INDIC_COUNT_NOUNS.values.flatten() +
            INDIC_TIMES_WORDS.values.flatten() + INDIC_EVERY_WORDS.values.flatten() +
            INDIC_INTERVAL_WORDS.values.flatten()
}
