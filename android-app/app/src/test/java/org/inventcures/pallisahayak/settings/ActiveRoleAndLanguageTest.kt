package org.inventcures.pallisahayak.settings

import com.google.common.truth.Truth.assertThat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.runBlocking
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.mockwebserver.RecordedRequest
import org.inventcures.pallisahayak.data.AskViewModel
import org.inventcures.pallisahayak.data.auth.ApiClients
import org.inventcures.pallisahayak.data.auth.StoredSession
import org.inventcures.pallisahayak.data.auth.TokenStore
import org.inventcures.pallisahayak.data.local.InteractionDao
import org.inventcures.pallisahayak.data.local.InteractionEntity
import org.inventcures.pallisahayak.data.local.LanguageCount
import org.inventcures.pallisahayak.data.model.CareRole
import org.inventcures.pallisahayak.data.model.Language
import org.inventcures.pallisahayak.data.prefs.InMemoryPreferencesStore
import org.inventcures.pallisahayak.data.release.ReleaseGate
import org.inventcures.pallisahayak.data.repository.PalliSahayakRepository
import org.inventcures.pallisahayak.data.repository.SessionContext
import org.inventcures.pallisahayak.data.settings.ActiveSettings
import org.inventcures.pallisahayak.data.start.AppStart
import org.inventcures.pallisahayak.data.start.StartScreen
import org.junit.After
import org.junit.Before
import org.junit.Test
import java.util.concurrent.CopyOnWriteArrayList

/**
 * The settings screen's Active Role and language switches, through the repository.
 *
 * The fake API is a MockWebServer answering like the mobile router. The rest is the
 * phone's real wiring: registration, the signed client, the repository with its Dose
 * Boundary, the Ask ViewModel and the settings it reads. What is checked is what a
 * researcher sees: the interaction row the phone records, and the language the
 * question went to the server in.
 */
class ActiveRoleAndLanguageTest {

    private lateinit var server: MockWebServer
    private lateinit var fake: FakeQueryServer
    private lateinit var clients: ApiClients
    private lateinit var rows: RecordingInteractions
    private lateinit var repository: PalliSahayakRepository
    private lateinit var prefs: InMemoryPreferencesStore
    private val work = SupervisorJob()

    @Before
    fun setUp() {
        fake = FakeQueryServer()
        server = MockWebServer().apply {
            dispatcher = fake
            start()
        }
        clients = ApiClients(baseUrl = server.url("/").toString(), tokens = MemoryTokens())
        rows = RecordingInteractions()
        repository = PalliSahayakRepository(api = clients.api, interactions = rows)
        prefs = InMemoryPreferencesStore()
    }

    @After
    fun tearDown() {
        work.cancel()
        server.shutdown()
    }

    private fun registerAs(role: CareRole, language: Language = Language.ENGLISH): SessionContext =
        runBlocking {
            clients.sessions.register(role = role, pin = "4821", language = language, siteId = "site-1")
            val start = AppStart(clients.authApi, ReleaseGate(developmentBuild = false), clients.sessions)
            (start.decide() as StartScreen.Ready).session
        }

    private fun askViewModel(session: SessionContext, settings: ActiveSettings) = AskViewModel(
        repository = repository,
        session = session,
        settings = settings,
        scope = CoroutineScope(Dispatchers.IO + work),
    )

    /** Ask through the screen's ViewModel, and wait for the row to be written. */
    private fun AskViewModel.ask(question: String) {
        onDraftChanged(question)
        submit()
        runBlocking { work.children.toList().joinAll() }
    }

    @Test
    fun `switching the Active Role changes what the next interaction records`() {
        val settings = ActiveSettings(prefs)
        val vm = askViewModel(registerAs(CareRole.ASHA_WORKER), settings)

        vm.ask("what helps with breathlessness?")
        settings.switchRole(CareRole.PATIENT)
        vm.ask("what helps with breathlessness?")

        assertThat(rows.inserted.map { it.careRole }).containsExactly("asha_worker", "patient").inOrder()
        // The same participant: switching hands over the phone, not the registration.
        assertThat(rows.inserted.map { it.participantId }.toSet()).containsExactly(fake.userId)
    }

    @Test
    fun `an unswitched phone records the Care Role it registered with`() {
        val vm = askViewModel(registerAs(CareRole.FAMILY_CAREGIVER), ActiveSettings(prefs))

        vm.ask("how do I turn him in bed?")

        assertThat(rows.inserted.single().careRole).isEqualTo("family_caregiver")
    }

    @Test
    fun `switching the Active Role sends it with the next question so the server records it`() {
        val settings = ActiveSettings(prefs)
        val vm = askViewModel(registerAs(CareRole.FAMILY_CAREGIVER), settings)

        vm.ask("how do I turn him in bed?")
        settings.switchRole(CareRole.ASHA_WORKER)
        vm.ask("how do I turn him in bed?")

        // The server's study row takes its speaker role from this field (ticket 25).
        assertThat(fake.queryActiveRoles).containsExactly("family_caregiver", "asha_worker").inOrder()
    }

    @Test
    fun `a spoken question carries the Active Role too`() {
        val settings = ActiveSettings(prefs)
        val session = registerAs(CareRole.ASHA_WORKER)
        settings.switchRole(CareRole.PATIENT)

        runBlocking { repository.askByVoice(byteArrayOf(0, 1, 2, 3), settings.applyTo(session)) }

        assertThat(fake.voiceActiveRoles).containsExactly("patient")
    }

    @Test
    fun `the Active Role is remembered across launches`() {
        val session = registerAs(CareRole.ASHA_WORKER)
        ActiveSettings(prefs).switchRole(CareRole.FAMILY_CAREGIVER)

        val relaunched = ActiveSettings(prefs)
        askViewModel(session, relaunched).ask("how do I turn him in bed?")

        assertThat(relaunched.activeRole(session)).isEqualTo(CareRole.FAMILY_CAREGIVER)
        assertThat(rows.inserted.single().careRole).isEqualTo("family_caregiver")
    }

    @Test
    fun `switching the language asks and records the next question in it`() {
        val settings = ActiveSettings(prefs)
        val vm = askViewModel(registerAs(CareRole.ASHA_WORKER, Language.ENGLISH), settings)

        vm.ask("what helps with breathlessness?")
        settings.switchLanguage(Language.HINDI)
        vm.ask("सांस फूलने में क्या मदद करता है?")

        assertThat(fake.queryLanguages).containsExactly("en-IN", "hi-IN").inOrder()
        assertThat(rows.inserted.map { it.language }).containsExactly("en-IN", "hi-IN").inOrder()
    }

    @Test
    fun `Odia is sent in the Supported Language table's code, not od-IN`() {
        val settings = ActiveSettings(prefs)
        val vm = askViewModel(registerAs(CareRole.PATIENT, Language.ODIA), settings)

        vm.ask("ଶ୍ୱାସ ପାଇଁ କଣ କରିବି?")

        assertThat(fake.registrationLanguages).containsExactly("or-IN")
        assertThat(fake.queryLanguages).containsExactly("or-IN")
        assertThat(rows.inserted.single().language).isEqualTo("or-IN")
    }

    @Test
    fun `a phone registered in od-IN by an older build asks in or-IN`() {
        val older = registerAs(CareRole.PATIENT).copy(language = "od-IN")

        askViewModel(older, ActiveSettings(prefs)).ask("ଶ୍ୱାସ ପାଇଁ କଣ କରିବି?")

        assertThat(fake.queryLanguages).containsExactly("or-IN")
    }
}

/** Answers registration, the release status and queries like `mobile_api/router.py`. */
private class FakeQueryServer : Dispatcher() {
    val userId = "3f2a8c14-9b7e-4d21-8a55-6c1e0d3b7f92"
    val queryLanguages = CopyOnWriteArrayList<String>()
    val registrationLanguages = CopyOnWriteArrayList<String>()
    val queryActiveRoles = CopyOnWriteArrayList<String>()
    val voiceActiveRoles = CopyOnWriteArrayList<String>()

    override fun dispatch(request: RecordedRequest): MockResponse {
        val body = request.body.readUtf8()
        return when (request.path.orEmpty().substringBefore('?')) {
            "/api/mobile/v1/release/status" -> json(
                """{"release_id":"rel-7","release_name":null,"status":"activated","activated":true,""" +
                    """"approved":true,"usable_with_participants":true,"drifting":false,"drift":[]}""",
            )
            "/api/mobile/v1/auth/register" -> {
                registrationLanguages += languageIn(body)
                json("""{"user_id":"$userId","token":"access-1","expires_at":1.7E9,"refresh_token":"refresh-1"}""")
            }
            "/api/mobile/v1/query/voice" -> {
                voiceActiveRoles += request.requestUrl?.queryParameter("active_role").orEmpty()
                json(
                    """{"transcript":"how do I turn him in bed?","answer":"Turn him every two hours.",""" +
                        """"sources":[],"evidence_level":"B","emergency_level":"none","confidence":0.8,""" +
                        """"validation_status":"validated","disclaimer":null,"audio_base64":null}""",
                )
            }
            "/api/mobile/v1/query" -> {
                queryLanguages += languageIn(body)
                queryActiveRoles += Regex("\"active_role\":\"([^\"]*)\"").find(body)?.groupValues?.get(1).orEmpty()
                json(
                    """{"answer":"Sit the patient upright and open a window.","sources":[],""" +
                        """"evidence_level":"B","emergency_level":"none","confidence":0.8,""" +
                        """"validation_status":"validated","disclaimer":null}""",
                )
            }
            else -> MockResponse().setResponseCode(404)
        }
    }

    private fun languageIn(body: String): String =
        Regex("\"language\":\"([^\"]*)\"").find(body)?.groupValues?.get(1).orEmpty()

    private fun json(body: String) = MockResponse()
        .setHeader("Content-Type", "application/json")
        .setBody(body)
}

private class MemoryTokens : TokenStore {
    @Volatile private var saved: StoredSession? = null
    override fun load(): StoredSession? = saved
    override fun save(session: StoredSession) {
        saved = session
    }
    override fun clear() {
        saved = null
    }
}

/** The interaction log, kept in memory so the test reads the rows the phone records. */
private class RecordingInteractions : InteractionDao {
    val inserted = CopyOnWriteArrayList<InteractionEntity>()
    override suspend fun insert(row: InteractionEntity) {
        inserted += row
    }
    override suspend fun recent(limit: Int): List<InteractionEntity> = inserted.take(limit)
    override fun observeRecent(): Flow<List<InteractionEntity>> = flowOf(inserted.toList())
    override suspend fun participantsWithActivitySince(since: Long): Int = 0
    override suspend fun substantiveSince(since: Long): List<InteractionEntity> = emptyList()
    override suspend fun dosageBlockedCount(): Int = 0
    override suspend fun emergencyCount(): Int = 0
    override suspend fun callsByLanguage(): List<LanguageCount> = emptyList()
    override suspend fun pruneBefore(before: Long) = Unit
}
