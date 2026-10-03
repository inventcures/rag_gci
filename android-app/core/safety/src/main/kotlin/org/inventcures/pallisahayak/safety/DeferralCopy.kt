package org.inventcures.pallisahayak.safety

/**
 * Deferral and redaction copy.
 *
 * Mirrors the server-side strings so the two agree. The wording is not decoration:
 * it must name what to do next, and it must never state or imply that a clinician
 * has been contacted. Protocol §7.6 forbids implying a transfer succeeded before
 * receipt is confirmed, and no notification path exists, so a promise here would be
 * a promise the system cannot keep.
 */
object DeferralCopy {

    private const val EN = "I understand this is what you need right now, and I am not able to help " +
        "with the specific quantity or schedule of a medicine — deciding that needs someone who can " +
        "see the patient.\n\n" +
        "What I can do is help you prepare for that conversation:\n" +
        "• Write down the symptoms, when they started, and what has already been given\n" +
        "• Bring the medicine packets or prescription along\n" +
        "• Raise it at the next consultation — with the palliative care doctor, the primary health " +
        "centre, or the OPD you are registered with\n\n" +
        "If the person is in severe distress right now — struggling to breathe, unresponsive, or in " +
        "pain that cannot be settled — call 108 for an ambulance rather than waiting for the next visit."

    private const val HI = "मैं समझ सकता हूँ कि यह अभी आपको ज़रूरी है। लेकिन किसी दवा की खास मात्रा या समय-सारणी " +
        "मैं नहीं बता सकता — यह तय करने के लिए मरीज़ को देखने वाले चिकित्सक की ज़रूरत है।\n\n" +
        "मैं इस बातचीत की तैयारी में मदद कर सकता हूँ:\n" +
        "• लक्षण, उन्हें कब शुरू हुए, और अब तक क्या दिया गया — लिख लें\n" +
        "• दवा के पाउच या प्रिस्क्रिप्शन साथ ले जाएँ\n" +
        "• अगली बार डॉक्टर से मिलने पर यह बात रखें — पैलिएटिव केयर डॉक्टर, प्राथमिक स्वास्थ्य " +
        "केंद्र, या जिस OPD में पंजीकरण है\n\n" +
        "अगर मरीज़ को अभी बहुत तेज़ तकलीफ़ है — साँस लेने में दिक्कत, बेहोशी, या न ठीक होने " +
        "वाला दर्द — तो अगली बार का इंतज़ार न करें, 108 पर एम्बुलेंस बुलाएँ।"

    private const val BN = "আমি বুঝতে পারছি এটি এখন আপনার দরকার। তবে কোনো ওষুধের নির্দিষ্ট পরিমাণ বা সময়সূচি " +
        "আমি বলতে পারি না — সেটা ঠিক করতে রোগীকে দেখেন এমন চিকিৎসক দরকার।\n\n" +
        "আমি এই কথোপকথনের প্রস্তুতিতে সাহায্য করতে পারি:\n" +
        "• লক্ষণ, কখন শুরু হয়েছে, এখন পর্যন্ত কী দেওয়া হয়েছে — লিখে রাখুন\n" +
        "• ওষুধের প্যাকেট বা প্রেসক্রিপশন সঙ্গে নিন\n" +
        "• পরবর্তী সাক্ষাতে ডাক্তারের সঙ্গে এ বিষয়ে কথা বলুন — প্যালিয়েটিভ কেয়ার ডাক্তার, " +
        "প্রাথমিক স্বাস্থ্য কেন্দ্র, বা আপনার নিবন্ধিত OPD\n\n" +
        "রোগী এখন খুব কষ্টে থাকলে — শ্বাসকষ্ট, অচেতন, বা সামলানো যায় না এমন ব্যথা — " +
        "পরের দিনের জন্য অপেক্ষা না করে 108 নম্বরে অ্যাম্বুলেন্স ডাকুন।"

    private const val TA = "இது இப்போது உங்களுக்கு தேவை என்று புரிகிறது. ஆனால் மருந்தின் குறிப்பிட்ட அளவு அல்லது " +
        "நேரத்தை நான் தெரிவிக்க முடியாது — அதைத் தீர்மானிக்க நோயாளியைப் பார்க்கும் மருத்துவர் தேவை.\n\n" +
        "இந்தப் பேச்சுக்கு தயார்படுத உதவலாம்:\n" +
        "• அறிகுறிகள், எப்போது தொடங்கியது, இதுவரை என்ன கொடுக்கப்பட்டது — எழுதி வையுங்கள்\n" +
        "• மருந்துப் பொதி அல்லது மருந்துச்சான்று எடுத்துச் செல்லுங்கள்\n" +
        "• அடுத்த சந்திப்பில் மருத்துவரிடம் இதை முன்வைக்கவும் — பாலியேட்டிவ் கேர் மருத்துவர், " +
        "முதல்நிலை சுகாதார மையம், அல்லது நீங்கள் பதிவு செய்த OPD\n\n" +
        "நோயாளர் இப்போது மிகவும் வேதனையில் இருந்தால் — மூச்சுத் திணறல், நினைவிழப்பு, அல்லது " +
        "சமாளிக்க முடியாத வலி — அடுத்த விஜயத்தை காத்திருக்காமல் 108-ல் ஆம்புலன்ஸை அழைக்கவும்."

    private val BY_PRIMARY_LANGUAGE = mapOf(
        "en" to EN,
        "hi" to HI,
        "bn" to BN,
        "ta" to TA,
    )

    /**
     * Fall back to English rather than refusing to answer.
     *
     * A user whose language has no translated deferral is better served by a
     * deferral in a language they may read than by silence.
     */
    fun forLanguage(language: String?): String {
        val primary = language?.substringBefore('-')?.substringBefore('_')?.lowercase()
        return BY_PRIMARY_LANGUAGE[primary] ?: EN
    }

    /** Appended after a partial redaction so the silence is explained. */
    fun redactionNote(language: String?): String = when (language?.substringBefore('-')?.lowercase()) {
        "hi" -> "\n\n_(किसी भी दवा की खास मात्रा और समय-सारणी यहाँ नहीं दी गई है। यह सवाल अपनी अगली डॉक्टर की विज़िट में पूछें।)_"
        "bn" -> "\n\n_(ওষুধের নির্দিষ্ট পরিমাণ ও সময়সূচি এখানে দেওয়া হয়নি। এই প্রশ্নটি আপনার পরবর্তী ডাক্তারের সাক্ষাতে করুন।)_"
        "ta" -> "\n\n_(எந்த மருந்தின் குறிப்பிட்ட அளவு மற்றும் நேரம் இங்கே தரப்படவில்லை. இந்தக் கேள்வியை அடுத்த மருத்துவர் சந்திப்பில் கேளுங்கள்.)_"
        else -> "\n\n_(The specific amount and schedule of any medicine are not included here. " +
            "Take that part of your question to your next consultation.)_"
    }
}
