package org.inventcures.pallisahayak.data.local

import androidx.room.ColumnInfo
import androidx.room.Dao
import androidx.room.Entity
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.PrimaryKey
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

/**
 * A pre-answered question from the Cached Answer Bundle (ADR 0005).
 *
 * The bundle answers only what it was built to answer. That is a deliberate
 * limit, not an oversight: a semantic index on the phone would need the 568M
 * parameter encoder downloaded over a 2G link, which is 7 to 27 hours and will
 * not fit in a 2 GB device.
 *
 * Matching is by hash rather than by similarity, because offline there is no model
 * available to embed the question. The consequence is that a differently worded
 * question misses, and the app has to say so plainly instead of returning
 * something approximate.
 */
@Entity(tableName = "cached_answers")
data class CachedAnswerEntity(
    @PrimaryKey
    @ColumnInfo(name = "query_hash")
    val queryHash: String,

    val language: String,

    /** The question as a user would say it, in the supported language. */
    @ColumnInfo(name = "query_text")
    val queryText: String,

    val response: String,

    /**
     * Mirrors AnswerKind, so a cached refusal is never rendered as an answer.
     * The bundle builder had a dose question in it once and the guard stripped it
     * on the way out, which is worse than a miss.
     */
    @ColumnInfo(name = "answer_kind")
    val answerKind: String,

    @ColumnInfo(name = "bundle_version")
    val bundleVersion: String,

    @ColumnInfo(name = "fetched_at")
    val fetchedAt: Long,
)

/**
 * Data Access Object: the only place in the app allowed to write SQL.
 *
 * Everything above this interface works in Kotlin objects. Swapping SQLite for
 * something else would touch this file and nothing else.
 */
@Dao
interface InteractionDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(row: InteractionEntity)

    @Query("SELECT * FROM interactions ORDER BY occurred_at DESC LIMIT :limit")
    suspend fun recent(limit: Int): List<InteractionEntity>

    /**
     * Returned as a Flow so Compose redraws when the list changes.
     *
     * A Flow is a stream that emits on demand. The screen subscribes once and
     * gets a new value whenever the database changes, instead of the screen having
     * to remember to re-query.
     */
    @Query("SELECT * FROM interactions ORDER BY occurred_at DESC")
    fun observeRecent(): Flow<List<InteractionEntity>>

    @Query(
        "SELECT COUNT(DISTINCT participant_id) FROM interactions " +
            "WHERE interaction_class = 'substantive' AND occurred_at >= :since",
    )
    suspend fun participantsWithActivitySince(since: Long): Int

    /**
     * Protocol §4.1: sustained use means an interaction in at least two distinct
     * weeks of the preceding 28 days. SQLite has no week-of-year function, so the
     * distinct weeks are counted in Kotlin from the rows this query returns.
     */
    @Query(
        "SELECT * FROM interactions " +
            "WHERE interaction_class = 'substantive' AND occurred_at >= :since",
    )
    suspend fun substantiveSince(since: Long): List<InteractionEntity>

    @Query("SELECT COUNT(*) FROM interactions WHERE dosage_blocked = 1")
    suspend fun dosageBlockedCount(): Int

    @Query("SELECT COUNT(*) FROM interactions WHERE emergency_level != 'none'")
    suspend fun emergencyCount(): Int

    /** Protocol §4.4 requires every metric broken out by language. */
    @Query("SELECT language, COUNT(*) AS calls FROM interactions GROUP BY language")
    suspend fun callsByLanguage(): List<LanguageCount>

    @Query("DELETE FROM interactions WHERE occurred_at < :before")
    suspend fun pruneBefore(before: Long)
}

/** Row shape for the grouped language query. Room needs a concrete return type. */
data class LanguageCount(
    val language: String,
    val calls: Int,
)

@Dao
interface CachedAnswerDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(rows: List<CachedAnswerEntity>)

    /**
     * Exact-match lookup on the question hash. Returns null when the question is
     * not one of the anticipated ones, which the caller must report honestly
     * rather than substituting the nearest thing it has.
     */
    @Query(
        "SELECT * FROM cached_answers WHERE query_hash = :hash AND language = :language",
    )
    suspend fun find(hash: String, language: String): CachedAnswerEntity?

    @Query("SELECT DISTINCT bundle_version FROM cached_answers")
    suspend fun versions(): List<String>

    /**
     * Replace the bundle atomically enough for this purpose.
     *
     * A newer bundle's rows land first, then rows from older versions go, so a
     * failed download leaves the previous bundle usable instead of emptying the
     * table.
     */
    @Query("DELETE FROM cached_answers WHERE bundle_version != :keepVersion")
    suspend fun deleteOtherVersions(keepVersion: String)

    @Query("SELECT COUNT(*) FROM cached_answers WHERE language = :language")
    suspend fun countForLanguage(language: String): Int
}
