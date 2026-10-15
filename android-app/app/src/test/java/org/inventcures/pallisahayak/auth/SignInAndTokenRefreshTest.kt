package org.inventcures.pallisahayak.auth

import com.google.common.truth.Truth.assertThat
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.runBlocking
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.mockwebserver.RecordedRequest
import org.inventcures.pallisahayak.data.auth.ApiClients
import org.inventcures.pallisahayak.data.auth.AuthResult
import org.inventcures.pallisahayak.data.auth.AuthState
import org.inventcures.pallisahayak.data.auth.StoredSession
import org.inventcures.pallisahayak.data.auth.TokenStore
import org.inventcures.pallisahayak.data.local.InteractionDao
import org.inventcures.pallisahayak.data.local.InteractionEntity
import org.inventcures.pallisahayak.data.local.LanguageCount
import org.inventcures.pallisahayak.data.model.CareRole
import org.inventcures.pallisahayak.data.model.Language
import org.inventcures.pallisahayak.data.release.GateDecision
import org.inventcures.pallisahayak.data.release.ReleaseGate
import org.inventcures.pallisahayak.data.repository.AnswerSource
import org.inventcures.pallisahayak.data.repository.PalliSahayakRepository
import org.inventcures.pallisahayak.data.start.AppStart
import org.inventcures.pallisahayak.data.start.StartScreen
import org.inventcures.pallisahayak.safety.AnswerKind
import org.junit.After
import org.junit.Before
import org.junit.Test
import java.util.concurrent.CopyOnWriteArrayList

/**
 * Registration, the bearer token and its refresh, through the repository.
 *
 * The fake API is a MockWebServer that answers the way the server's mobile
 * router does (NOTES/reports/08: the refresh token goes in the body, an
 * access token is refused there, a missing or expired access token is a 401).
 * Everything on the phone's side is real: the Retrofit client generated from
 * the contract, the OkHttp interceptor that signs each call, the authenticator
 * that refreshes, the session manager and the repository with its Dose
 * Boundary. Only the interaction log is an in-memory stand-in for Room, which
 * is a storage boundary these tests do not look at.
 */
class SignInAndTokenRefreshTest {

    private lateinit var server: MockWebServer
    private lateinit var fake: FakeMobileServer
    private lateinit var tokens: MemoryTokenStore
    private lateinit var clients: ApiClients
    private lateinit var repository: PalliSahayakRepository

    @Before
    fun setUp() {
        fake = FakeMobileServer()
        server = MockWebServer().apply {
            dispatcher = fake
            start()
        }
        tokens = MemoryTokenStore()
        clients = ApiClients(baseUrl = server.url("/").toString(), tokens = tokens)
        repository = PalliSahayakRepository(api = clients.api, interactions = NoopInteractions())
    }

    @After
    fun tearDown() {
        server.shutdown()
    }

    private fun approvedStart() = AppStart(
        api = clients.authApi,
        gate = ReleaseGate(developmentBuild = false),
        sessions = clients.sessions,
    )

    private fun register(): AuthResult = runBlocking {
        clients.sessions.register(
            role = CareRole.ASHA_WORKER,
            pin = "4821",
            language = Language.HINDI,
            siteId = "site-1",
        )
    }

    private fun readySession() = runBlocking {
        (approvedStart().decide() as StartScreen.Ready).session
    }

    @Test
    fun `a registered ASHA Worker's question carries the bearer token and is answered`() {
        assertThat(register()).isEqualTo(AuthResult.SignedIn)

        val answered = runBlocking { repository.ask("what helps with breathlessness?", readySession()) }

        assertThat(answered.kind).isEqualTo(AnswerKind.ANSWER)
        assertThat(answered.source).isInstanceOf(AnswerSource.Live::class.java)
        assertThat(answered.text).contains("Sit the patient upright")
        assertThat(fake.queryAuthorizations).containsExactly("Bearer ${fake.issuedAccess.last()}")
    }

    @Test
    fun `registration sends the Care Role and PIN and no name or phone`() {
        register()

        val body = fake.registrations.single()
        assertThat(body).contains("\"role\":\"asha_worker\"")
        assertThat(body).contains("\"pin\":\"4821\"")
        assertThat(body).contains("\"language\":\"hi-IN\"")
        assertThat(body).contains("\"name\":\"\"")
        assertThat(body).contains("\"phone_hash\":\"\"")
    }

    @Test
    fun `the session the app runs on is the registered participant, not a placeholder`() {
        register()

        val session = readySession()

        assertThat(session.participantId).isEqualTo(fake.userId)
        assertThat(session.careRole).isEqualTo("asha_worker")
        assertThat(session.siteId).isEqualTo("site-1")
        assertThat(session.language).isEqualTo("hi-IN")
        assertThat(session.releaseId).isEqualTo("rel-7")
        assertThat(session.releaseApproved).isTrue()
    }

    @Test
    fun `an expired access token is refreshed once and the question is still answered`() {
        register()
        val session = readySession()
        fake.expireAccessTokens()

        val answered = runBlocking { repository.ask("what helps with breathlessness?", session) }

        assertThat(answered.kind).isEqualTo(AnswerKind.ANSWER)
        assertThat(fake.refreshBodies).hasSize(1)
        assertThat(fake.refreshBodies.single()).contains(fake.issuedRefresh.first())
        // The refresh call itself carries no bearer token (report 08).
        assertThat(fake.refreshAuthorizations).containsExactly(null as String?)
        assertThat(fake.queryAuthorizations.last()).isEqualTo("Bearer ${fake.issuedAccess.last()}")
        assertThat(clients.sessions.state.value).isInstanceOf(AuthState.SignedIn::class.java)
    }

    @Test
    fun `a failed refresh returns the participant to PIN sign-in`() {
        register()
        val session = readySession()
        fake.expireAccessTokens()
        fake.rejectRefresh = true

        val answered = runBlocking { repository.ask("what helps with breathlessness?", session) }

        assertThat(fake.refreshBodies).hasSize(1)
        assertThat(answered.kind).isEqualTo(AnswerKind.DEFERRAL)
        assertThat(clients.sessions.state.value).isEqualTo(AuthState.NeedsPin(fake.userId))
        assertThat(runBlocking { approvedStart().decide() }).isEqualTo(StartScreen.SignIn)
        // The device keeps who it belongs to, so the PIN is all that is needed.
        assertThat(tokens.saved?.userId).isEqualTo(fake.userId)
        assertThat(tokens.saved?.accessToken).isNull()
    }

    @Test
    fun `the PIN signs the participant back in after a failed refresh`() {
        register()
        val before = readySession()
        fake.expireAccessTokens()
        fake.rejectRefresh = true
        runBlocking { repository.ask("what helps with breathlessness?", before) }
        fake.rejectRefresh = false

        val result = runBlocking { clients.sessions.signIn("4821") }
        val answered = runBlocking { repository.ask("what helps with breathlessness?", readySession()) }

        assertThat(result).isEqualTo(AuthResult.SignedIn)
        assertThat(fake.logins.single()).contains("\"user_id\":\"${fake.userId}\"")
        assertThat(answered.kind).isEqualTo(AnswerKind.ANSWER)
    }

    @Test
    fun `a wrong PIN is refused and leaves the participant at sign-in`() {
        register()
        fake.expireAccessTokens()
        fake.rejectRefresh = true
        runBlocking { repository.ask("anything", readySession()) }

        val result = runBlocking { clients.sessions.signIn("0000") }

        assertThat(result).isEqualTo(AuthResult.Refused)
        assertThat(clients.sessions.state.value).isEqualTo(AuthState.NeedsPin(fake.userId))
    }

    @Test
    fun `starting against an unapproved release shows the refusal screen`() {
        fake.releaseApproved = false
        register()

        val screen = runBlocking { approvedStart().decide() }

        assertThat(screen).isInstanceOf(StartScreen.Refused::class.java)
        val blocked = (screen as StartScreen.Refused).decision
        assertThat(blocked.awaitingApproval).isTrue()
        assertThat(fake.queryAuthorizations).isEmpty()
    }

    @Test
    fun `starting against a release that cannot be checked shows the refusal screen`() {
        fake.releaseUnreachable = true

        val screen = runBlocking { approvedStart().decide() }

        assertThat(screen).isInstanceOf(StartScreen.Refused::class.java)
    }

    @Test
    fun `a first start on an approved release asks the participant to register`() {
        assertThat(runBlocking { approvedStart().decide() }).isEqualTo(StartScreen.Register)
    }

    @Test
    fun `an unapproved release is refused before registration is offered`() {
        fake.releaseApproved = false

        val screen = runBlocking { approvedStart().decide() }

        assertThat(screen).isInstanceOf(StartScreen.Refused::class.java)
        assertThat((screen as StartScreen.Refused).decision).isInstanceOf(GateDecision.Blocked::class.java)
    }
}

/**
 * Answers like `mobile_api/router.py` for the routes the app calls.
 *
 * Tokens are opaque strings here; what matters to the phone is which one the
 * server will accept, and that is the contract report 08 records.
 */
private class FakeMobileServer : Dispatcher() {
    val userId = "3f2a8c14-9b7e-4d21-8a55-6c1e0d3b7f92"
    private val pin = "4821"

    val issuedAccess = CopyOnWriteArrayList<String>()
    val issuedRefresh = CopyOnWriteArrayList<String>()
    val registrations = CopyOnWriteArrayList<String>()
    val logins = CopyOnWriteArrayList<String>()
    val refreshBodies = CopyOnWriteArrayList<String>()
    val refreshAuthorizations = CopyOnWriteArrayList<String?>()
    val queryAuthorizations = CopyOnWriteArrayList<String?>()

    @Volatile var rejectRefresh = false
    @Volatile var releaseApproved = true
    @Volatile var releaseUnreachable = false

    @Volatile private var validAccess: Set<String> = emptySet()

    fun expireAccessTokens() {
        validAccess = emptySet()
    }

    override fun dispatch(request: RecordedRequest): MockResponse {
        val path = request.path.orEmpty().substringBefore('?')
        val body = request.body.readUtf8()
        return when (path) {
            "/api/mobile/v1/release/status" -> releaseStatus()
            "/api/mobile/v1/auth/register" -> {
                registrations += body
                issuePair()
            }
            "/api/mobile/v1/auth/login" -> {
                logins += body
                if (body.contains("\"pin\":\"$pin\"")) issuePair() else unauthorized("Invalid credentials")
            }
            "/api/mobile/v1/auth/refresh" -> {
                refreshBodies += body
                refreshAuthorizations += request.getHeader("Authorization")
                val known = issuedRefresh.any { body.contains("\"refresh_token\":\"$it\"") }
                if (known && !rejectRefresh) issuePair() else unauthorized("Invalid or expired refresh token")
            }
            "/api/mobile/v1/query" -> {
                val authorization = request.getHeader("Authorization")
                queryAuthorizations += authorization
                val token = authorization?.removePrefix("Bearer ")
                if (token != null && token in validAccess) answer() else unauthorized("Invalid or expired token")
            }
            else -> MockResponse().setResponseCode(404)
        }
    }

    private fun issuePair(): MockResponse {
        val n = issuedAccess.size + 1
        val access = "access-$n"
        val refresh = "refresh-$n"
        issuedAccess += access
        issuedRefresh += refresh
        validAccess = validAccess + access
        return json(
            """{"user_id":"$userId","token":"$access","expires_at":1.7E9,"refresh_token":"$refresh"}""",
        )
    }

    private fun releaseStatus(): MockResponse {
        if (releaseUnreachable) return MockResponse().setResponseCode(503)
        val ok = releaseApproved
        return json(
            """{"release_id":"rel-7","release_name":null,"status":"${if (ok) "activated" else "draft"}",""" +
                """"activated":$ok,"approved":$ok,"usable_with_participants":$ok,""" +
                """"drifting":false,"drift":[]}""",
        )
    }

    private fun answer() = json(
        """{"answer":"Sit the patient upright and open a window. Call the nurse if it gets worse.",""" +
            """"sources":[{"filename":"handbook.pdf"}],"evidence_level":"B","emergency_level":"none",""" +
            """"confidence":0.8,"validation_status":"validated","disclaimer":null}""",
    )

    private fun unauthorized(detail: String) =
        json("""{"detail":"$detail"}""").setResponseCode(401)

    private fun json(body: String) = MockResponse()
        .setHeader("Content-Type", "application/json")
        .setBody(body)
}

/** Holds the session in memory, as the Keystore-backed store does on a phone. */
private class MemoryTokenStore : TokenStore {
    @Volatile var saved: StoredSession? = null

    override fun load(): StoredSession? = saved

    override fun save(session: StoredSession) {
        saved = session
    }

    override fun clear() {
        saved = null
    }
}

/** The interaction log, which these tests do not read. */
private class NoopInteractions : InteractionDao {
    override suspend fun insert(row: InteractionEntity) = Unit
    override suspend fun recent(limit: Int): List<InteractionEntity> = emptyList()
    override fun observeRecent(): Flow<List<InteractionEntity>> = flowOf(emptyList())
    override suspend fun participantsWithActivitySince(since: Long): Int = 0
    override suspend fun substantiveSince(since: Long): List<InteractionEntity> = emptyList()
    override suspend fun dosageBlockedCount(): Int = 0
    override suspend fun emergencyCount(): Int = 0
    override suspend fun callsByLanguage(): List<LanguageCount> = emptyList()
    override suspend fun pruneBefore(before: Long) = Unit
}
