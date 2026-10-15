package org.inventcures.pallisahayak.data.repository

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import okhttp3.RequestBody
import org.inventcures.pallisahayak.api.generated.AuthResponse
import org.inventcures.pallisahayak.api.generated.CacheBundleResponse
import org.inventcures.pallisahayak.api.generated.DeviceRegistrationRequest
import org.inventcures.pallisahayak.api.generated.LoginRequest
import org.inventcures.pallisahayak.api.generated.MobileQueryRequest
import org.inventcures.pallisahayak.api.generated.MobileQueryResponse
import org.inventcures.pallisahayak.api.generated.PalliSahayakApi
import org.inventcures.pallisahayak.api.generated.RefreshRequest
import org.inventcures.pallisahayak.api.generated.ReleaseStatusResponse
import org.inventcures.pallisahayak.api.generated.VoiceQueryResponse
import org.inventcures.pallisahayak.data.local.CachedAnswerDao
import org.inventcures.pallisahayak.data.local.CachedAnswerEntity
import org.inventcures.pallisahayak.data.local.InteractionDao
import org.inventcures.pallisahayak.data.local.InteractionEntity
import org.inventcures.pallisahayak.data.local.LanguageCount
import retrofit2.Response
import java.io.IOException

/**
 * The server, as the repository sees it. Set [offline] to make every call fail
 * the way a phone with no signal fails: with an IOException.
 */
class FakeApi : PalliSahayakApi {
    var offline = false
    var queryResponse: MobileQueryResponse? = null
    var voiceResponse: VoiceQueryResponse? = null
    var bundle: CacheBundleResponse? = null
    var release: ReleaseStatusResponse? = null
    val bundleRequests = mutableListOf<String>()

    fun answer(answer: String, emergencyLevel: String = "none") {
        queryResponse = MobileQueryResponse(
            answer = answer,
            confidence = 0.8,
            disclaimer = null,
            emergency_level = emergencyLevel,
            evidence_level = "B",
            sources = emptyList(),
            validation_status = "validated",
        )
    }

    fun voice(transcript: String, answer: String, emergencyLevel: String = "none") {
        voiceResponse = VoiceQueryResponse(
            answer = answer,
            audio_base64 = null,
            confidence = 0.8,
            disclaimer = null,
            emergency_level = emergencyLevel,
            evidence_level = "B",
            sources = emptyList(),
            transcript = transcript,
            validation_status = "validated",
        )
    }

    private fun <T> respond(body: T?): Response<T> {
        if (offline) throw IOException("no signal")
        return Response.success(body ?: error("the fake server has no response set"))
    }

    override suspend fun query(body: MobileQueryRequest): Response<MobileQueryResponse> = respond(queryResponse)

    override suspend fun voiceQuery(language: String, active_role: String?, audio: RequestBody): Response<VoiceQueryResponse> =
        respond(voiceResponse)

    override suspend fun cacheBundle(language: String): Response<CacheBundleResponse> {
        bundleRequests.add(language)
        return respond(bundle)
    }

    override suspend fun register(body: DeviceRegistrationRequest): Response<AuthResponse> = error("not used")

    override suspend fun login(body: LoginRequest): Response<AuthResponse> = error("not used")

    override suspend fun refresh(body: RefreshRequest): Response<AuthResponse> = error("not used")

    override suspend fun releaseStatus(): Response<ReleaseStatusResponse> = respond(release)
}

/** The interaction table, in memory. */
class FakeInteractionDao : InteractionDao {
    val rows = mutableListOf<InteractionEntity>()

    override suspend fun insert(row: InteractionEntity) {
        rows.removeAll { it.id == row.id }
        rows.add(row)
    }

    override suspend fun recent(limit: Int): List<InteractionEntity> =
        rows.sortedByDescending { it.occurredAt }.take(limit)

    override fun observeRecent(): Flow<List<InteractionEntity>> = MutableStateFlow(rows.toList())

    override suspend fun participantsWithActivitySince(since: Long): Int = error("not used")

    override suspend fun substantiveSince(since: Long): List<InteractionEntity> = error("not used")

    override suspend fun dosageBlockedCount(): Int = rows.count { it.dosageBlocked }

    override suspend fun emergencyCount(): Int = rows.count { it.emergencyLevel != "none" }

    override suspend fun callsByLanguage(): List<LanguageCount> = error("not used")

    override suspend fun pruneBefore(before: Long) {
        rows.removeAll { it.occurredAt < before }
    }
}

/** The Cached Answer Bundle table, in memory. */
class FakeCachedAnswerDao : CachedAnswerDao {
    val rows = mutableListOf<CachedAnswerEntity>()

    override suspend fun insertAll(rows: List<CachedAnswerEntity>) {
        for (row in rows) {
            this.rows.removeAll { it.queryHash == row.queryHash }
            this.rows.add(row)
        }
    }

    override suspend fun find(hash: String, language: String): CachedAnswerEntity? =
        rows.firstOrNull { it.queryHash == hash && it.language == language }

    override suspend fun versions(): List<String> = rows.map { it.bundleVersion }.distinct()

    override suspend fun deleteOtherVersions(keepVersion: String) {
        rows.removeAll { it.bundleVersion != keepVersion }
    }

    override suspend fun deleteStale(language: String, keepVersion: String) {
        rows.removeAll { it.language == language && it.bundleVersion != keepVersion }
    }

    override suspend fun countForLanguage(language: String): Int = rows.count { it.language == language }
}
