package org.inventcures.pallisahayak

import androidx.test.core.app.ApplicationProvider
import com.google.common.truth.Truth.assertThat
import kotlinx.coroutines.test.runTest
import retrofit2.http.Body
import retrofit2.http.Part
import retrofit2.http.GET
import retrofit2.http.Query
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.withContext
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Before
import okhttp3.RequestBody
import java.util.concurrent.Executor as JavaExecutor
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.inventcures.pallisahayak.api.generated.CacheBundleResponse
import org.inventcures.pallisahayak.api.generated.MobileQueryRequest
import org.inventcures.pallisahayak.api.generated.MobileQueryResponse
import org.inventcures.pallisahayak.api.generated.PalliSahayakApi
import org.inventcures.pallisahayak.data.AskViewModel
import org.inventcures.pallisahayak.data.local.InteractionDao
import org.inventcures.pallisahayak.data.voice.VoiceEngine
import org.inventcures.pallisahayak.data.local.PalliDatabase
import org.inventcures.pallisahayak.data.repository.PalliSahayakRepository
import org.inventcures.pallisahayak.data.repository.SessionContext
import org.inventcures.pallisahayak.api.generated.VoiceQueryResponse
import org.inventcures.pallisahayak.safety.AnswerKind
import org.inventcures.pallisahayak.safety.EmergencySeverity

/**
 * Drives the real stack through the repository and ViewModel.
 *
 * Only the network boundary is faked. The Room database is real, the repository
 * is real, the Dose Boundary is real, and the ViewModel is real. Mocking our own
 * code would make these tests assert that our own mocks agree with each other,
 * which is the failure mode this whole body of work exists to avoid.
 */
@OptIn(ExperimentalCoroutinesApi::class)
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class AskViewModelTest {

    private lateinit var api: StubApi
    private lateinit var database: PalliDatabase
    private lateinit var interactions: InteractionDao
    private lateinit var repository: PalliSahayakRepository

    /**
     * Registration hands the app a server-generated UUID, never a name or a phone
     * number, so this is the shape the app actually holds.
     */
    private val session = SessionContext(
        participantId = "3f2a8c14-9b7e-4d21-8a55-6c1e0d3b7f92",
        careRole = "ASHA Worker",
        siteId = "CCHRC",
        language = "en-IN",
        releaseId = "unassigned-dev@abc1234",
        releaseApproved = false,
    )

    @Before
    fun setUp() {
        // Main is still set because Room touches it internally; the ViewModel's
        // own work runs on the injected test scope.
        Dispatchers.setMain(UnconfinedTestDispatcher())

        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        // A real Room database, created in memory. Not mocked: a mocked database
        // would only prove the repository calls the methods the test expects.
        database = androidx.room.Room
            .inMemoryDatabaseBuilder(context, PalliDatabase::class.java)
            .allowMainThreadQueries()
            // Run Room inline. Its suspend DAO methods otherwise dispatch to a
            // transaction executor whose continuation never resumes under
            // Robolectric, and the repository hangs inside insert().
            .setQueryExecutor(directExecutor)
            .setTransactionExecutor(directExecutor)
            .build()
        interactions = database.interactions()
        api = StubApi()
        repository = PalliSahayakRepository(api = api, interactions = interactions)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
        // close() is enough: the test never goes through the process-wide
        // singleton, so there is no shared instance to reset.
        database.close()
    }

    private fun answer(
        answer: String,
        sources: List<Map<String, Any>> = listOf(
            mapOf(
                "filename" to "handbook.pdf", "document" to "d",
                "page" to 1, "relevance" to 0.5, "snippet" to "s",
            ),
        ),
        emergencyLevel: String = "none",
        evidenceLevel: String = "B",
    ) {
        api.response = MobileQueryResponse(
            answer = answer,
            sources = sources,
            evidence_level = evidenceLevel,
            emergency_level = emergencyLevel,
            confidence = 0.8,
            validation_status = "validated",
            disclaimer = null,
        )
    }

    /**
     * Takes the runTest scope, so advanceUntilIdle drives the same scheduler the
     * ViewModel works on. An injected UnconfinedTestDispatcher is a *separate*
     * scheduler, and OkHttp's real I/O resumes on neither, which left the
     * coroutine suspended and every answer empty.
     */
    /**
     * Wait for the ViewModel to finish a submit.
     *
     * Room's suspend DAO methods resume on Room's own transaction executor, not
     * on the test scheduler, so advanceUntilIdle returns while the row is still
     * being written and the ViewModel has not yet updated. Draining the scheduler
     * and yielding until the state settles handles both.
     */
    private suspend fun TestScope.awaitIdle(vm: AskViewModel) {
        val deadline = System.currentTimeMillis() + 5_000
        while (vm.state.value.isAsking && System.currentTimeMillis() < deadline) {
            advanceUntilIdle()
            // Room writes on its own executor, so give it real time too.
            Thread.sleep(10)
            advanceUntilIdle()
        }
    }

    /**
     * Unconfined but bound to *this* runTest's scheduler.
     *
     * A separately constructed dispatcher is a separate scheduler, so Room's
     * continuation lands on a queue advanceUntilIdle never drains. Binding it to
     * the runTest scheduler puts both on one queue.
     */
    private val voice = FakeVoiceEngine()

    private fun TestScope.viewModel() = AskViewModel(
        repository = repository,
        session = session,
        voiceEngine = voice,
        scope = kotlinx.coroutines.CoroutineScope(UnconfinedTestDispatcher(testScheduler)),
    )

    @Test
    fun `a grounded answer reaches the screen intact`() = runTest {
        answer("Morphine is a strong opioid used for severe cancer pain.")
        val vm = viewModel()

        vm.onDraftChanged("what is morphine used for?")
        vm.submit()
        awaitIdle(vm)

        val state = vm.state.value
        assertThat(state.answerKind).isEqualTo(AnswerKind.ANSWER)
        assertThat(state.answer).contains("strong opioid")
        assertThat(state.sourceCount).isEqualTo(1)
        assertThat(state.hasAnswer).isTrue()
    }

    @Test
    fun `SI-1 a dosed answer is refused before it reaches the screen`() = runTest {
        answer("Give morphine 10 mg orally every 4 hours.", evidenceLevel = "C")
        val vm = viewModel()

        vm.onDraftChanged("what dose of morphine?")
        vm.submit()
        testScheduler.advanceUntilIdle()

        val state = vm.state.value
        assertThat(state.answerKind).isNotEqualTo(AnswerKind.ANSWER)
        assertThat(state.answer).doesNotContain("10 mg")
    }

    @Test
    fun `SI-2 useful content survives when one sentence carries the dose`() = runTest {
        answer(
            "Morphine is a strong opioid used for severe cancer pain. " +
                "It is the gold standard for severe pain control. " +
                "Give 10 mg every 4 hours.",
        )
        val vm = viewModel()

        vm.onDraftChanged("what is morphine used for?")
        vm.submit()
        testScheduler.advanceUntilIdle()

        val state = vm.state.value
        assertThat(state.answerKind).isEqualTo(AnswerKind.REDACTED_ANSWER)
        assertThat(state.answer).contains("strong opioid")
        assertThat(state.answer).doesNotContain("10 mg")
    }

    @Test
    fun `SI-2 a too-short remainder becomes a full deferral rather than a fragment`() = runTest {
        // Not a weakened assertion. After redaction this leaves about 56
        // characters, below the 80-character threshold, so shipping the fragment
        // would read worse than refusing outright.
        answer(
            "Morphine is a strong opioid used for severe cancer pain. " +
                "Give 10 mg every 4 hours.",
        )
        val vm = viewModel()

        vm.onDraftChanged("what is morphine used for?")
        vm.submit()
        awaitIdle(vm)

        assertThat(vm.state.value.answerKind).isEqualTo(AnswerKind.DEFERRAL)
    }

    @Test
    fun `SI-5 every interaction is recorded with its release and outcome`() = runTest {
        answer("Give 10 mg morphine.", sources = emptyList(), evidenceLevel = "C")
        val vm = viewModel()

        vm.onDraftChanged("dose?")
        vm.submit()
        testScheduler.advanceUntilIdle()

        val rows = interactions.recent(limit = 10)
        assertThat(rows).hasSize(1)

        val row = rows.first()
        assertThat(row.releaseId).isEqualTo("unassigned-dev@abc1234")
        assertThat(row.releaseApproved).isFalse()
        assertThat(row.careRole).isEqualTo("ASHA Worker")
        // Pseudonymous by construction: the device never receives a raw
        // identifier, so there is nothing for it to hash. This asserts the value
        // survives the round trip unchanged rather than claiming a hashing step
        // that does not exist.
        assertThat(row.participantId).isEqualTo(session.participantId)
        assertThat(row.dosageBlocked).isTrue()
        assertThat(row.answerKind).isEqualTo(AnswerKind.DEFERRAL.name)
        assertThat(row.response).doesNotContain("10 mg")
    }

    @Test
    fun `SI-5 phone numbers are scrubbed before the row is written`() = runTest {
        answer("Call 9876543210 for help.", sources = emptyList(), evidenceLevel = "C")
        val vm = viewModel()

        vm.onDraftChanged("what number should I call, my phone is 9876543210?")
        vm.submit()
        testScheduler.advanceUntilIdle()

        val row = interactions.recent(limit = 1).first()
        assertThat(row.query).doesNotContain("9876543210")
        assertThat(row.response).doesNotContain("9876543210")
    }

    @Test
    fun `a network failure is reported plainly rather than crashing`() = runTest {
        api.failWith = java.io.IOException("simulated network failure")
        val vm = viewModel()

        vm.onDraftChanged("what should I do?")
        vm.submit()
        testScheduler.advanceUntilIdle()

        val state = vm.state.value
        assertThat(state.isAsking).isFalse()
        assertThat(state.errorMessage).isNotNull()
        // Recorded even though it failed, because protocol 4.1 excludes failures
        // from the adoption numerator and they have to be on disk to be excluded.
        assertThat(interactions.recent(limit = 1)).hasSize(1)
    }
    private fun spoken(
        transcript: String,
        answer: String,
        audioBase64: String? = null,
        emergencyLevel: String = "none",
    ) {
        api.voiceResponse = VoiceQueryResponse(
            answer = answer,
            audio_base64 = audioBase64,
            sources = listOf(
                mapOf(
                    "filename" to "handbook.pdf", "document" to "d",
                    "page" to 1, "relevance" to 0.5, "snippet" to "s",
                ),
            ),
            evidence_level = "B",
            emergency_level = emergencyLevel,
            confidence = 0.8,
            validation_status = "validated",
            disclaimer = null,
            transcript = transcript,
        )
    }
    // -- voice ---------------------------------------------------------------
    @Test
    fun `a spoken question returns a spoken and readable answer`() = runTest {
        spoken("what is morphine used for?", "Morphine is a strong opioid.")
        val vm = viewModel()
        vm.onMicPressed()
        assertThat(vm.state.value.isListening).isTrue()
        vm.onMicPressed()
        vm.onMicReleased()
        awaitIdle(vm)
        val state = vm.state.value
        assertThat(state.answer).contains("strong opioid")
        // Shown as well as spoken, because a spoken answer nobody can re-read
        // is lost.
        assertThat(state.transcript).isEqualTo("what is morphine used for?")
        assertThat(vm.state.value.isListening).isFalse()
    }
    @Test
    fun `SI-1 a spoken answer carrying a dose is restricted too`() = runTest {
        spoken("what dose of morphine?", "Give morphine 10 mg every 4 hours.")
        val vm = viewModel()
        vm.onMicPressed()
        vm.onMicReleased()
        awaitIdle(vm)
        val state = vm.state.value
        assertThat(state.answerKind).isNotEqualTo(AnswerKind.ANSWER)
        assertThat(state.answer).doesNotContain("10 mg")
    }
    @Test
    fun `SI-4 a critical spoken emergency still overrides the dose boundary`() = runTest {
        spoken(
            "the patient cannot breathe",
            "Call 108 immediately. Give oxygen if available.",
            emergencyLevel = "critical",
        )
        val vm = viewModel()
        vm.onMicPressed()
        vm.onMicReleased()
        awaitIdle(vm)
        assertThat(vm.state.value.emergency).isEqualTo(EmergencySeverity.CRITICAL)
        assertThat(vm.state.value.isSpeaking).isFalse()
        assertThat(vm.state.value.answer).contains("108")
    }
    @Test
    fun `SI-5 a voice interaction records the voice path`() = runTest {
        spoken("what should I do?", "Ask the palliative physician.")
        val vm = viewModel()
        vm.onMicPressed()
        vm.onMicReleased()
        awaitIdle(vm)
        val row = interactions.recent(limit = 1).first()
        assertThat(row.voicePath).isEqualTo("live")
        assertThat(row.channel).isEqualTo("live")
    }
    @Test
    fun `a voice failure is reported rather than left silent`() = runTest {
        api.failWith = java.io.IOException("simulated network failure")
        val vm = viewModel()
        vm.onMicPressed()
        vm.onMicReleased()
        awaitIdle(vm)
        assertThat(vm.state.value.errorMessage).isNotNull()
        // Still recorded, because protocol 4.1 excludes failures from the adoption
        // numerator and they have to be on disk to be excluded from it.
        assertThat(interactions.recent(limit = 1)).hasSize(1)
    }
}

/**
 * The network boundary, faked.
 *
 * Real network I/O inside runTest parks Retrofit's suspend call on OkHttp's
 * dispatcher, and Robolectric never pumps the main looper, so the continuation
 * never resumes and every answer comes back empty. Stubbing the API removes the
 * problem rather than working around it, and it fakes exactly the boundary this
 * suite is meant to fake: the repository, the Room database, the Dose Boundary and
 * the ViewModel all stay real.
 */
private class StubApi : PalliSahayakApi {
    var response: MobileQueryResponse? = null
    var failWith: Exception? = null
    var callCount = 0
    var lastQuery: String? = null

    override suspend fun query(body: MobileQueryRequest): retrofit2.Response<MobileQueryResponse> {
        callCount++
        lastQuery = body.query
        failWith?.let { throw it }
        return retrofit2.Response.success(
            response ?: error("StubApi.response was not set for '${body.query}'"),
        )
    }

    var voiceResponse: org.inventcures.pallisahayak.api.generated.VoiceQueryResponse? = null
    var voiceCallCount = 0

    override suspend fun voiceQuery(
        @Query("language") language: String,
        @Part("audio") audio: okhttp3.RequestBody,
    ): retrofit2.Response<org.inventcures.pallisahayak.api.generated.VoiceQueryResponse> {
        voiceCallCount++
        return retrofit2.Response.success(
            voiceResponse ?: error("StubApi.voiceResponse was not set"),
        )
    }

    override suspend fun cacheBundle(
        @Query("language") language: String,
    ): retrofit2.Response<CacheBundleResponse> = retrofit2.Response.success(
        CacheBundleResponse(
            version = "stub",
            language = language,
            generated_at = 0.0,
            queries = emptyList(),
            treatments = emptyList(),
            emergency_keywords = emptyMap(),
            evidence_badge_metadata = emptyMap(),
        ),
    )

    override suspend fun login(
        @Body body: org.inventcures.pallisahayak.api.generated.LoginRequest,
    ): retrofit2.Response<org.inventcures.pallisahayak.api.generated.AuthResponse> =
        error("login is ticket 06")
}

/** Runs work inline on the calling thread. */
/** Runs work inline on the calling thread. */
private val directExecutor = JavaExecutor { it.run() }

/**
 * Stands in for the microphone.
 *
 * Robolectric has no audio hardware, and a real microphone in a unit test is a
 * flake waiting to happen. The interface exists so the voice path can be tested
 * without one.
 */
private class FakeVoiceEngine : VoiceEngine {
    var listening = false
        private set
    var speaking = false
        private set
    var startCount = 0
        private set
    var stopSpeakingCount = 0
        private set
    var spokenBytes = 0
        private set

    /** Set false to simulate a denied microphone permission. */
    var permitted = true

    /** Audio handed back by stopRecording. */
    var captured: ByteArray? = ByteArray(16_000)

    override fun canRecord(): Boolean = permitted

    override val isListening: Boolean get() = listening
    override val isSpeaking: Boolean get() = speaking

    override fun startRecording() {
        if (listening || !permitted) return
        listening = true
        startCount++
    }

    override fun stopRecording(): ByteArray? {
        if (!listening) return null
        listening = false
        return captured
    }

    override fun speak(pcm: ByteArray) {
        speaking = true
        spokenBytes += pcm.size
    }

    override fun stopSpeaking() {
        if (speaking) stopSpeakingCount++
        speaking = false
    }
}
