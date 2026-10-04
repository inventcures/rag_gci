package org.inventcures.pallisahayak.data.offline

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import org.inventcures.pallisahayak.data.local.InteractionDao
import org.inventcures.pallisahayak.data.local.InteractionEntity
import org.inventcures.pallisahayak.safety.AnswerKind
import org.inventcures.pallisahayak.safety.EmergencySeverity

/**
 * One past interaction, as the history list renders it.
 *
 * `kind` is carried through rather than re-derived. A refusal saved three weeks ago
 * was judged by the boundary at the time it was given; re-running the guard on it now
 * would judge the copy, not the answer, and could disagree with what was said then.
 */
data class HistoryEntry(
    val id: String,
    val question: String,
    val answer: String,
    val kind: AnswerKind,
    val emergency: EmergencySeverity,
    val wasOffline: Boolean,
    val cameFromCache: Boolean,
    val occurredAt: Long,
)

/**
 * Reads past interactions, with no network.
 *
 * Criterion: history is readable offline and survives the app being closed. That is
 * satisfied by reading the local table, and the durability is Room's rather than
 * anything this class does. What this class is responsible for is not quietly losing
 * a kind: a Redacted Answer from weeks ago must still render differently from an
 * answer, or someone rereads a refusal as clinical advice.
 */
class HistoryRepository(private val dao: InteractionDao) {

    fun observe(): Flow<List<HistoryEntry>> =
        dao.observeRecent().map { rows -> rows.map { it.toEntry() } }

    suspend fun recent(limit: Int = 50): List<HistoryEntry> =
        dao.recent(limit).map { it.toEntry() }

    /**
     * Whether anything is readable offline.
     *
     * Used by the offline screen to tell "nothing recorded yet" apart from "nothing
     * can be shown", which lead to different behaviour and confuse a worker.
     */
    fun hasHistory(): Flow<Boolean> = dao.observeRecent().map { it.isNotEmpty() }

    private fun InteractionEntity.toEntry() = HistoryEntry(
        id = id,
        question = query,
        answer = response,
        kind = kindOrDefault(),
        emergency = EmergencySeverity.fromLabel(emergencyLevel),
        wasOffline = isOffline,
        cameFromCache = fromCache,
        occurredAt = occurredAt,
    )

    /**
     * An unrecognised kind reads as an answer, not a refusal.
     *
     * The fallback is the safe direction here. An older build may have recorded a
     * kind this build does not know, and showing that text with answer styling risks
     * a refusal being read as advice. Showing it as a refusal risks the opposite,
     * which is merely unhelpful.
     */
    private fun InteractionEntity.kindOrDefault(): AnswerKind =
        runCatching { AnswerKind.valueOf(answerKind) }.getOrDefault(AnswerKind.ANSWER)
}