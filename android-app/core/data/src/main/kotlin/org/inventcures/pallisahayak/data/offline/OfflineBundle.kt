package org.inventcures.pallisahayak.data.offline

import android.util.Log
import com.squareup.moshi.JsonDataException
import org.inventcures.pallisahayak.api.generated.PalliSahayakApi
import org.inventcures.pallisahayak.data.local.CachedAnswerDao
import org.inventcures.pallisahayak.data.local.CachedAnswerEntity
import org.inventcures.pallisahayak.data.model.Language
import org.inventcures.pallisahayak.safety.DoseBoundary
import java.io.IOException

/**
 * Downloads the Cached Answer Bundle for one language while there is signal.
 *
 * ADR 0005: offline means a downloaded bundle of anticipated answers. Nothing on
 * the phone fetched one before ticket 23, so the offline path could never answer.
 * A newer bundle's rows land first and the language's older rows go after, so a
 * failed download leaves the previous bundle usable.
 *
 * Each row's [CachedAnswerEntity.answerKind] is what the client Dose Boundary
 * makes of the answer on arrival. The boundary runs again when the answer is
 * shown; storing the kind means a refused answer is never shown as an answer,
 * even if the two readings ever differed.
 */
class OfflineBundleDownloader(
    private val api: PalliSahayakApi,
    private val dao: CachedAnswerDao,
    private val boundary: DoseBoundary = DoseBoundary(),
    private val clock: () -> Long = System::currentTimeMillis,
) {

    /**
     * Fetch and store [language]'s bundle. Returns how many answers were stored,
     * or null when there was nothing new to store (no signal, a server error, or
     * the bundle already held, which the server answers with 304).
     */
    suspend fun refresh(language: Language): Int? {
        val response = try {
            api.cacheBundle(language.tag)
        } catch (error: IOException) {
            Log.w(TAG, "bundle download failed; keeping the bundle already held", error)
            return null
        } catch (error: JsonDataException) {
            Log.w(TAG, "bundle could not be read; keeping the bundle already held", error)
            return null
        }
        val body = response.body()
        if (!response.isSuccessful || body == null) return null

        val fetchedAt = clock()
        val rows = body.queries.mapNotNull { entry ->
            val hash = entry["query_hash"] as? String
            val answer = entry["response_text"] as? String
            if (hash.isNullOrBlank() || answer.isNullOrBlank()) return@mapNotNull null
            CachedAnswerEntity(
                queryHash = hash,
                language = language.tag,
                queryText = entry["query_text"] as? String ?: "",
                response = answer,
                answerKind = boundary.apply(answer, language.tag).kind.name,
                bundleVersion = body.version,
                fetchedAt = fetchedAt,
            )
        }
        // An empty bundle (the server could not draft answers) must not empty the
        // one the phone already holds.
        if (rows.isEmpty()) return 0
        dao.insertAll(rows)
        dao.deleteStale(language.tag, body.version)
        return rows.size
    }

    private companion object {
        const val TAG = "PalliBundle"
    }
}
