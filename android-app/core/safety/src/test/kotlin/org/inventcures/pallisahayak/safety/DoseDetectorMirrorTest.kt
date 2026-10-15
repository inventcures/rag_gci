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

    @Test
    fun `stage IV is not read as the IV route`() {
        val answer = "Stage IV cancer can still be cared for at home with good support."
        assertThat(DoseDetector.containsDose(answer)).isFalse()
    }
}
