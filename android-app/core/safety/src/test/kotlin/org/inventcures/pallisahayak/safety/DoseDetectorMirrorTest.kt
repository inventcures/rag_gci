package org.inventcures.pallisahayak.safety

import com.google.common.truth.Truth.assertThat
import com.google.common.truth.Truth.assertWithMessage
import org.junit.jupiter.api.Test

/**
 * The client Dose Boundary catches every dose shape the server's does.
 *
 * Round 3 (ticket 01) rewrote the server detector, `dose_detector.py`, to catch
 * spelled-out units, doses in words, counts of tablets and puffs, "4-hourly",
 * Hindi "एमजी", and a dose beside an age. The Android mirror missed all of them,
 * matched Indic schedule words inside longer words ("बार" in "बारे"), and carried
 * unit words that mixed scripts, so they could never match. The rows below are
 * the server's own rows from `tests/test_dose_boundary_shapes.py`.
 */
class DoseDetectorMirrorTest {

    private val boundary = DoseBoundary()

    private fun assertDoseWithheld(answer: String, language: String, dose: String) {
        val result = boundary.apply(answer, language = language)
        assertWithMessage("blocked: $answer").that(result.isDoseRestricted).isTrue()
        assertThat(result.text).doesNotContain(dose)
    }

    @Test
    fun `every review B3 miss is withheld`() {
        val rows = listOf(
            Triple("Morphine 5 milligrams is what the hospice used.", "en-IN", "5 milligrams"),
            Triple("The patch gives 25 micrograms per hour of fentanyl.", "en-IN", "25 micrograms"),
            Triple("Paracetamol 500 milligrammes TDS is usual.", "en-IN", "500 milligrammes"),
            Triple("Give five milligrams every four hours.", "en-IN", "five milligrams"),
            Triple("Take two tablets at night.", "en-IN", "two tablets"),
            Triple("Use 2 puffs of the inhaler when breathless.", "en-IN", "2 puffs"),
            Triple("Oral morphine is usually given 4-hourly.", "en-IN", "4-hourly"),
            Triple("मॉर्फिन 5 एमजी दें।", "hi-IN", "5 एमजी"),
            Triple("मॉर्फिन 10 एम.जी. दें।", "hi-IN", "10 एम.जी."),
            Triple("For patients aged 70, give morphine 10 mg every 4 hours.", "en-IN", "10 mg"),
        )
        for ((answer, language, dose) in rows) assertDoseWithheld(answer, language, dose)
    }

    @Test
    fun `the unit word of every supported language is withheld`() {
        val rows = listOf(
            Triple("Give the patient 10 milligrams of morphine.", "en-IN", "10 milligrams"),
            Triple("फेंटानिल पैच से 25 माइक्रोग्राम मिलता है।", "hi-IN", "25 माइक्रोग्राम"),
            Triple("रुग्णाला 10 मिलिग्रॅम मॉर्फिन द्या.", "mr-IN", "10 मिलिग्रॅम"),
            Triple("রোগীকে ৫ গ্রাম প্যারাসিটামল দিন।", "bn-IN", "৫ গ্রাম"),
            Triple("ਮਰੀਜ਼ ਨੂੰ 5 ਗ੍ਰਾਮ ਦਵਾਈ ਦਿਓ।", "pa-IN", "5 ਗ੍ਰਾਮ"),
            Triple("દર્દીને 10 મિલિગ્રામ મોર્ફિન આપો.", "gu-IN", "10 મિલિગ્રામ"),
            Triple("ରୋଗୀଙ୍କୁ 10 ମିଲିଗ୍ରାମ ମର୍ଫିନ ଦିଅନ୍ତୁ।", "od-IN", "10 ମିଲିଗ୍ରାମ"),
            Triple("രോഗിക്ക് 10 മില്ലിഗ്രാം മോർഫിൻ നൽകുക.", "ml-IN", "10 മില്ലിഗ്രാം"),
            Triple("நோயாளிக்கு 10 மில்லிகிராம் மார்பின் கொடுங்கள்.", "ta-IN", "10 மில்லிகிராம்"),
            Triple("రోగికి 10 మిల్లీగ్రాములు మార్ఫిన్ ఇవ్వండి.", "te-IN", "10 మిల్లీగ్రాములు"),
            Triple("ರೋಗಿಗೆ 10 ಮಿಲಿಗ್ರಾಂ ಮಾರ್ಫಿನ್ ನೀಡಿ.", "kn-IN", "10 ಮಿಲಿಗ್ರಾಂ"),
        )
        for ((answer, language, dose) in rows) assertDoseWithheld(answer, language, dose)
    }

    @Test
    fun `the age exception spares only the age, not the dose beside it`() {
        val answer = "Pain is common in people aged 70 and over, and it can be eased. " +
            "Keep a diary of when the pain comes and what helps. " +
            "For patients aged 70, give morphine 10 mg every 4 hours."
        val result = boundary.apply(answer, language = "en-IN")
        assertThat(result.kind).isEqualTo(AnswerKind.REDACTED_ANSWER)
        assertThat(result.text).contains("aged 70 and over")
        assertThat(result.text).doesNotContain("10 mg")
    }

    @Test
    fun `a word containing a frequency word passes untouched`() {
        val rows = listOf(
            "परिवार को बीमारी के बारे में बताएँ।" to "hi-IN",
            "दिनेश घर पर आराम कर रहे हैं।" to "hi-IN",
            "घरात आराम करा." to "mr-IN",
            "ਪਰਿਵਾਰ ਨਾਲ ਗੱਲ ਕਰੋ।" to "pa-IN",
            "તહેવાર પર પરિવાર સાથે રહો." to "gu-IN",
            "ಅವರು ಒಬ್ಬರೇ ಇರಬಾರದು." to "kn-IN",
            "முறையாக ஓய்வு எடுங்கள்." to "ta-IN",
            "రోజా పువ్వు అందంగా ఉంది." to "te-IN",
            "ସରୋଜ ଘରେ ଅଛନ୍ତି।" to "od-IN",
        )
        for ((answer, language) in rows) {
            val result = boundary.apply(answer, language = language)
            assertWithMessage("findings for $answer").that(result.doseFindings).isEmpty()
            assertThat(result.text).isEqualTo(answer)
        }
    }

    @Test
    fun `a schedule word, whole or inflected, is still withheld`() {
        val rows = listOf(
            "मॉर्फिन दिन में दो बार दें।" to "hi-IN",
            "यह दवा रोज़ सुबह दें।" to "hi-IN",
            "ਇਹ ਦਵਾਈ ਦੋ ਵਾਰ ਦਿਓ।" to "pa-IN",
            "ಈ ಔಷಧಿಯನ್ನು ಪ್ರತಿ ದಿನ ನೀಡಿ." to "kn-IN",
            "यह दवा रोज़ाना दें।" to "hi-IN",
            "यह दवा प्रतिदिन दें।" to "hi-IN",
            "रातों को यह दवा दें।" to "hi-IN",
            "मरीज़ को दो गोलियों से आराम मिला।" to "hi-IN",
            "मरीज़ को दो खुराकें दें।" to "hi-IN",
            "रात्री ही दवा द्या." to "mr-IN",
            "ಈ ಔಷಧಿಯನ್ನು ಎರಡು ಬಾರಿ ನೀಡಿ." to "kn-IN",
            "ಈ ಔಷಧಿಯನ್ನು ದಿನಕ್ಕೆ ನೀಡಿ." to "kn-IN",
            "ఈ మందు రోజుకు ఇవ్వండి." to "te-IN",
            "ఈ మందు మూడు రోజులు ఇవ్వండి." to "te-IN",
            "இந்த மருந்தை தினமும் கொடுங்கள்." to "ta-IN",
            "இந்த மருந்தை வாரத்திற்கு கொடுங்கள்." to "ta-IN",
            "ਇਹ ਦਵਾਈ ਰੋਜ਼ਾਨਾ ਦਿਓ।" to "pa-IN",
            "આ દવા દરરોજ આપો." to "gu-IN",
            "આ દવા દિવસમાં આપો." to "gu-IN",
            "ଏହି ଔଷଧ ପ୍ରତିଦିନ ଦିଅନ୍ତୁ।" to "od-IN",
            "ଏହି ଔଷଧ ଦିନକୁ ଦିଅନ୍ତୁ।" to "od-IN",
        )
        for ((answer, language) in rows) {
            val result = boundary.apply(answer, language = language)
            assertWithMessage("blocked: $answer").that(result.isDoseRestricted).isTrue()
            // The Deferral itself may use the same word ("अगली बार"), so check
            // that the schedule sentence is gone, not the word.
            assertThat(result.text).doesNotContain(answer)
        }
    }

    /**
     * The server's COUNTS_AND_SCHEDULES_IN_EACH_LANGUAGE rows (final review,
     * round 3, finding 1): before them, counts were English and Hindi only.
     */
    @Test
    fun `a count or schedule in each language is withheld`() {
        val rows = listOf(
            Triple("दर चार तासांनी दोन गोळ्या द्या", "mr-IN", "दोन गोळ्या"),
            Triple("இரண்டு மாத்திரைகளை ஒவ்வொரு நான்கு மணி நேரத்திற்கும் கொடுக்கவும்", "ta-IN", "இரண்டு மாத்திரைகளை"),
            Triple("દર ચાર કલાકે બે ગોળી આપો", "gu-IN", "બે ગોળી"),
            Triple("প্রতি চার ঘণ্টা অন্তর দুটি ট্যাবলেট দিন", "bn-IN", "দুটি ট্যাবলেট"),
            Triple("காலையில் 2 மாத்திரை சாப்பிடுங்கள்.", "ta-IN", "2 மாத்திரை"),
            Triple("দিনে দুবার খান", "bn-IN", "দুবার"),
            Triple("दिवसातून दोनदा घ्या", "mr-IN", "दोनदा"),
            Triple("ਦੋ ਗੋਲੀਆਂ ਲਓ", "pa-IN", "ਦੋ ਗੋਲੀਆਂ"),
            Triple("Take ½ tablet", "en-IN", "½ tablet"),
            Triple("Take 1½ tablets at night.", "en-IN", "1½ tablets"),
            Triple("Give ¼ tablet when the pain starts.", "en-IN", "¼ tablet"),
            Triple("Use ¾ teaspoon of the syrup.", "en-IN", "¾ teaspoon"),
            Triple("½ गोली दें।", "hi-IN", "½ गोली"),
            Triple("हर चार घंटे में दवा दें।", "hi-IN", "हर चार घंटे"),
            Triple("রাতে ২টি ট্যাবলেট খান।", "bn-IN", "২টি ট্যাবলেট"),
            Triple("দিনে তিন বার এই ওষুধ দিন।", "bn-IN", "তিন বার"),
            Triple("প্রতি ৬ ঘণ্টায় ওষুধ দিন।", "bn-IN", "প্রতি ৬ ঘণ্টায়"),
            Triple("চোখে দুই ফোঁটা দিন।", "bn-IN", "দুই ফোঁটা"),
            Triple("ওষুধটি চার ঘণ্টা পর পর দিন।", "bn-IN", "চার ঘণ্টা পর পর"),
            Triple("दिवसातून तीन वेळा औषध द्या.", "mr-IN", "तीन वेळा"),
            Triple("जेवणानंतर 2 गोळ्या द्या.", "mr-IN", "2 गोळ्या"),
            Triple("डोळ्यात दोन थेंब टाका.", "mr-IN", "दोन थेंब"),
            Triple("दर ६ तासांनी औषध द्या.", "mr-IN", "दर ६ तासांनी"),
            Triple("જમ્યા પછી બે વખત દવા આપો.", "gu-IN", "બે વખત"),
            Triple("દર 6 કલાકે દવા આપો.", "gu-IN", "દર 6 કલાકે"),
            Triple("એક ચમચી સીરપ આપો.", "gu-IN", "એક ચમચી"),
            Triple("અડધી ગોળી આપો.", "gu-IN", "અડધી ગોળી"),
            Triple("ਹਰ ਚਾਰ ਘੰਟੇ ਬਾਅਦ ਦਵਾਈ ਦਿਓ।", "pa-IN", "ਹਰ ਚਾਰ ਘੰਟੇ"),
            Triple("ਅੱਧੀ ਗੋਲੀ ਦਿਓ।", "pa-IN", "ਅੱਧੀ ਗੋਲੀ"),
            Triple("ਅੱਖ ਵਿੱਚ 2 ਬੂੰਦਾਂ ਪਾਓ।", "pa-IN", "2 ਬੂੰਦਾਂ"),
            Triple("ஒரு நாளைக்கு மூன்று தடவை கொடுக்கவும்.", "ta-IN", "மூன்று தடவை"),
            Triple("இரண்டு சொட்டு கண்ணில் விடவும்.", "ta-IN", "இரண்டு சொட்டு"),
            Triple("அரை மாத்திரை கொடுக்கவும்.", "ta-IN", "அரை மாத்திரை"),
            Triple("இரவில் இரண்டு வேளை கொடுக்கவும்.", "ta-IN", "இரண்டு வேளை"),
            Triple("రెండు మాత్రలు వేసుకోండి.", "te-IN", "రెండు మాత్రలు"),
            Triple("ప్రతి నాలుగు గంటలకు మందు ఇవ్వండి.", "te-IN", "ప్రతి నాలుగు గంటలకు"),
            Triple("నాలుగు గంటలకోసారి మందు ఇవ్వండి.", "te-IN", "నాలుగు గంటలకోసారి"),
            Triple("అర మాత్ర ఇవ్వండి.", "te-IN", "అర మాత్ర"),
            Triple("ಎರಡು ಮಾತ್ರೆಗಳನ್ನು ನೀಡಿ.", "kn-IN", "ಎರಡು ಮಾತ್ರೆಗಳನ್ನು"),
            Triple("ಪ್ರತಿ ನಾಲ್ಕು ಗಂಟೆಗೆ ಔಷಧಿ ನೀಡಿ.", "kn-IN", "ಪ್ರತಿ ನಾಲ್ಕು ಗಂಟೆಗೆ"),
            Triple("ನಾಲ್ಕು ಗಂಟೆಗೊಮ್ಮೆ ಔಷಧಿ ನೀಡಿ.", "kn-IN", "ನಾಲ್ಕು ಗಂಟೆಗೊಮ್ಮೆ"),
            Triple("ಮೂರು ಸಲ ಔಷಧಿ ನೀಡಿ.", "kn-IN", "ಮೂರು ಸಲ"),
            Triple("രണ്ട് ഗുളിക കഴിക്കുക.", "ml-IN", "രണ്ട് ഗുളിക"),
            Triple("ഓരോ നാല് മണിക്കൂറിലും മരുന്ന് നൽകുക.", "ml-IN", "ഓരോ നാല് മണിക്കൂറിലും"),
            Triple("ദിവസം മൂന്നു നേരം മരുന്ന് നൽകുക.", "ml-IN", "മൂന്നു നേരം"),
            Triple("ദിവസം രണ്ട് തവണ മരുന്ന് നൽകുക.", "ml-IN", "രണ്ട് തവണ"),
            Triple("കണ്ണിൽ രണ്ടു തുള്ളി ഒഴിക്കുക.", "ml-IN", "രണ്ടു തുള്ളി"),
            Triple("ଦୁଇଟି ଟାବଲେଟ ଦିଅନ୍ତୁ।", "od-IN", "ଦୁଇଟି ଟାବଲେଟ"),
            Triple("ପ୍ରତି ଚାରି ଘଣ୍ଟାରେ ଔଷଧ ଦିଅନ୍ତୁ।", "od-IN", "ପ୍ରତି ଚାରି ଘଣ୍ଟାରେ"),
            Triple("ତିନି ଥର ଔଷଧ ଦିଅନ୍ତୁ।", "od-IN", "ତିନି ଥର"),
            // "দেড়" (one and a half) with the nukta letter as two code points.
            Triple("দেড় ট্যাবলেট দিন।", "bn-IN", "ট্যাবলেট"),
        )
        for ((answer, language, dose) in rows) assertDoseWithheld(answer, language, dose)
    }

    @Test
    fun `an ordinary sentence with a count word passes untouched`() {
        val rows = listOf(
            "রোগীকে জল দিন।" to "bn-IN",
            "দুই ঘণ্টা বিশ্রাম নিন।" to "bn-IN",
            "আমি আপনাকে দুটি কথা বলি।" to "bn-IN",
            "चार तासांनी परत या." to "mr-IN",
            "दोन गोष्टी लक्षात ठेवा." to "mr-IN",
            "બે દીકરા ઘરે છે." to "gu-IN",
            "ਦੋ ਗੱਲਾਂ ਯਾਦ ਰੱਖੋ।" to "pa-IN",
            "ஒரு வேளை வலி அதிகமானால் மருத்துவரை அழைக்கவும்." to "ta-IN",
            "நான்கு மணிக்கு மருத்துவர் வருவார்." to "ta-IN",
            "రెండు మాత్రమే మిగిలాయి." to "te-IN",
            "ಎರಡು ಸಲಹೆಗಳನ್ನು ನೆನಪಿಡಿ." to "kn-IN",
            "രണ്ട് കുട്ടികൾ വീട്ടിലുണ്ട്." to "ml-IN",
            "കുറച്ച് നേരം വിശ്രമിക്കുക." to "ml-IN",
            "ଦୁଇ ଜଣ ଘରେ ଅଛନ୍ତି।" to "od-IN",
            "About ½ of patients feel tired." to "en-IN",
        )
        for ((answer, language) in rows) {
            val result = boundary.apply(answer, language = language)
            assertWithMessage("findings for $answer").that(result.doseFindings).isEmpty()
            assertThat(result.text).isEqualTo(answer)
        }
    }

    @Test
    fun `stage IV is not read as the IV route`() {
        val answer = "Stage IV cancer can still be cared for at home with good support."
        assertThat(DoseDetector.containsDose(answer)).isFalse()
    }
}
