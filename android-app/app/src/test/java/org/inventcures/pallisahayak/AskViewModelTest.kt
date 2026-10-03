package org.inventcures.pallisahayak

import androidx.test.core.app.ApplicationProvider
import com.google.common.truth.Truth.assertThat
import kotlinx.coroutines.test.runTest
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.inventcures.pallisahayak.data.AskViewModel
import org.inventcures.pallisahayak.data.ServiceLocator
import org.inventcures.pallisahayak.data.local.InteractionDao
import org.inventcures.pallisahayak.data.local.PalliDatabase
import org.inventcures.pallisahayak.data.repository.PalliSahayakRepository
import org.inventcures.pallisahayak.data.repository.SessionContext
import org.inventcures.pallisahayak.safety.AnswerKind

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

    private lateinit var server: MockWebServer
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
        // Main is still set because Room and Retrofit touch it internally, but the
        // ViewModel's own work runs on the injected test scope.
        // Unconfined, not Standard. runTest creates its own scheduler, so a
        // StandardTestDispatcher installed here would be a *second* one, and
        // advancing testScheduler would drain the wrong queue. Unconfined runs
        // the launch eagerly on the calling thread, which is what a
        // ViewModel-under-test needs.
        Dispatchers.setMain(UnconfinedTestDispatcher())
        server = MockWebServer()
        server.start()

        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        // A real Room database, created in memory. The database is not mocked,
        // because a mocked database would only prove the repository calls the
        // methods the test expects it to call.
        database = androidx.room.Room
            .inMemoryDatabaseBuilder(context, PalliDatabase::class.java)
            .allowMainThreadQueries()
            .build()
        interactions = database.interactions()

        val client = okhttp3.OkHttpClient.Builder().build()
        val api = retrofit2.Retrofit.Builder()
            .baseUrl(server.url("/api/mobile/v1/"))
            .client(client)
            .addConverterFactory(
                retrofit2.converter.moshi.MoshiConverterFactory.create(
                    com.squareup.moshi.Moshi.Builder()
                        .add(com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory())
                        .build(),
                ),
            )
            .build()
            .create(org.inventcures.pallisahayak.api.generated.PalliSahayakApi::class.java)

        repository = PalliSahayakRepository(api = api, interactions = interactions)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
        server.shutdown()
        // close() is enough: the test never goes through the process-wide
        // singleton, so there is no shared instance to reset.
        database.close()
    }

    private fun answer(body: String) {
        server.enqueue(
            MockResponse()
                .setResponseCode(200)
                .setHeader("Content-Type", "application/json")
                .setBody(body),
        )
    }

    /**
     * Takes the runTest scope, so advanceUntilIdle drives the same scheduler the
     * ViewModel works on. An injected UnconfinedTestDispatcher is a *separate*
     * scheduler, and OkHttp's real I/O resumes on neither, which left the
     * coroutine suspended and every answer empty.
     */
    private fun TestScope.viewModel() = AskViewModel(repository, session, this)

    @Test
    fun `a grounded answer reaches the screen intact`() = runTest {
        answer(
            """{"answer":"Morphine is a strong opioid used for severe cancer pain.",
               "sources":[{"filename":"handbook.pdf"}],"evidence_level":"B",
               "emergency_level":"none","confidence":0.8,
               "validation_status":"validated","disclaimer":null}""",
        )
        val vm = viewModel()

        vm.onDraftChanged("what is morphine used for?")
        vm.submit()
        testScheduler.advanceUntilIdle()

        val state = vm.state.value
        assertThat(state.answerKind).isEqualTo(AnswerKind.ANSWER)
        assertThat(state.answer).contains("strong opioid")
        assertThat(state.sourceCount).isEqualTo(1)
        assertThat(state.hasAnswer).isTrue()
    }

    @Test
    fun `SI-1 a dosed answer is refused before it reaches the screen`() = runTest {
        answer(
            """{"answer":"Give morphine 10 mg orally every 4 hours.",
               "sources":[{"filename":"handbook.pdf"}],"evidence_level":"C",
               "emergency_level":"none","confidence":0.6,
               "validation_status":"validated","disclaimer":null}""",
        )
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
            """{"answer":"Morphine is a strong opioid used for severe cancer pain. Give 10 mg every 4 hours.",
               "sources":[{"filename":"handbook.pdf"}],"evidence_level":"B",
               "emergency_level":"none","confidence":0.7,
               "validation_status":"validated","disclaimer":null}""",
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
    fun `SI-5 every interaction is recorded with its release and outcome`() = runTest {
        answer(
            """{"answer":"Give 10 mg morphine.","sources":[],"evidence_level":"C",
               "emergency_level":"none","confidence":0.5,
               "validation_status":"validated","disclaimer":null}""",
        )
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
        answer(
            """{"answer":"Call 9876543210 for help.","sources":[],"evidence_level":"C",
               "emergency_level":"none","confidence":0.5,
               "validation_status":"validated","disclaimer":null}""",
        )
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
        server.enqueue(MockResponse().setResponseCode(503))
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
}
