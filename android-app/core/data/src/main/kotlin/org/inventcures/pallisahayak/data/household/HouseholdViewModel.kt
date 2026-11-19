package org.inventcures.pallisahayak.data.household

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import org.inventcures.pallisahayak.data.local.HouseholdMemberEntity
import org.inventcures.pallisahayak.data.model.Language
import org.inventcures.pallisahayak.data.reminders.DailyReminder
import org.inventcures.pallisahayak.data.reminders.ReminderRepository
import org.inventcures.pallisahayak.data.repository.SessionContext
import org.inventcures.pallisahayak.data.settings.ActiveSettings
import org.inventcures.pallisahayak.safety.DoseDetector
import java.util.UUID

/**
 * The symptoms a carer can record, by the server's names for them
 * (`memory_agents/ingest_agent.py`). A short fixed list: the screen has to be
 * usable by someone who reads little, and a typed symptom is free text.
 */
enum class Symptom(val wire: String) {
    PAIN("pain"),
    BREATHLESSNESS("breathlessness"),
    NAUSEA("nausea"),
    CONSTIPATION("constipation"),
}

/** How bad a symptom is, on the server's scale (`SeverityLevel`, 0 to 4). */
enum class Severity(val wire: Int) {
    NONE(0),
    MILD(1),
    MODERATE(2),
    SEVERE(3),
    VERY_SEVERE(4),
}

/**
 * What the household screen has to tell the participant. A kind, not a sentence:
 * the screen words it in their language.
 */
enum class HouseholdNotice {
    /** Stored on the phone, and waiting for a connection to reach the record. */
    OBSERVATION_SAVED,

    /** Nothing was stored. Said plainly, so nobody believes it was recorded. */
    OBSERVATION_FAILED,
    REMINDER_SAVED,

    /** The medicine's name held an amount. A reminder never carries a dose. */
    REMINDER_HAS_DOSE,

    /** Something else failed; nothing was changed. */
    FAILED,
}

data class HouseholdUiState(
    val members: List<HouseholdMemberEntity> = emptyList(),
    val reminders: List<DailyReminder> = emptyList(),
    val notice: HouseholdNotice? = null,
    /** False until the first read, so the screen does not flash "nobody yet". */
    val loaded: Boolean = false,
)

/**
 * The people a participant cares for, what was noticed about them, and their
 * medicine reminders.
 *
 * Names stay on the phone. The server knows a person only by a random reference
 * made here when they are added; the first observation pushed under it makes
 * that record this participant's (`claim_or_check` on the server).
 */
class HouseholdViewModel(
    private val households: HouseholdRepository,
    private val reminders: ReminderRepository,
    private val session: SessionContext,
    /** The language in force, as for a question. Null only in tests without settings. */
    private val settings: ActiveSettings? = null,
    /** Injected for tests, as in AskViewModel. */
    private val scope: CoroutineScope? = null,
    private val newPatientRef: () -> String = { UUID.randomUUID().toString() },
) : ViewModel() {

    private val workScope: CoroutineScope
        get() = scope ?: viewModelScope

    private val owner: String get() = session.participantId

    private val _state = MutableStateFlow(HouseholdUiState())
    val state: StateFlow<HouseholdUiState> = _state.asStateFlow()

    init {
        act(onSuccess = null) { }
    }

    /** Add a person by the name the family uses. The household is made on the first. */
    fun addMember(name: String) {
        if (name.isBlank()) return
        act(onSuccess = null) {
            val household = households.register(owner, session.siteId, language(), session.releaseId)
            households.addMember(owner, household.id, name, newPatientRef())
        }
    }

    fun recordObservation(memberId: String, symptom: Symptom, severity: Severity) {
        workScope.launch {
            val saved = attempt {
                households.recordObservation(owner, memberId, Observation(symptom.wire, severity.wire))
            } ?: false
            _state.update {
                it.copy(
                    notice = if (saved) HouseholdNotice.OBSERVATION_SAVED else HouseholdNotice.OBSERVATION_FAILED,
                )
            }
        }
    }

    /**
     * Remind every day at [minutesOfDay].
     *
     * [medicine] is what the carer typed and [text] is the sentence the screen
     * built around it, in the participant's language. A name that carries an
     * amount ("morphine 10 mg") is refused: the Dose Boundary that applies to an
     * answer applies to a reminder, which a family acts on without anyone to check.
     */
    fun addReminder(medicine: String, text: String, minutesOfDay: Int) {
        if (medicine.isBlank()) return
        if (DoseDetector.containsDose(medicine)) {
            _state.update { it.copy(notice = HouseholdNotice.REMINDER_HAS_DOSE) }
            return
        }
        act(onSuccess = HouseholdNotice.REMINDER_SAVED) {
            reminders.addDaily(text, minutesOfDay, language().tag)
        }
    }

    fun removeReminder(medicationRef: String) {
        act(onSuccess = null) { reminders.remove(medicationRef) }
    }

    fun clearNotice() {
        _state.update { it.copy(notice = null) }
    }

    private fun language(): Language =
        settings?.language?.value ?: Language.fromTag(session.language) ?: Language.ALL.first()

    /** Run [change], then show what is stored now, whether or not it worked. */
    private fun act(onSuccess: HouseholdNotice?, change: suspend () -> Unit) {
        workScope.launch {
            val done = attempt { change(); true } ?: false
            val members = attempt { households.members(owner) }
            val daily = attempt { reminders.daily() }
            _state.update {
                it.copy(
                    members = members ?: it.members,
                    reminders = daily ?: it.reminders,
                    notice = if (done) onSuccess else HouseholdNotice.FAILED,
                    loaded = true,
                )
            }
        }
    }

    /** Null when [block] fails. A full disk or a refused write must not crash a visit. */
    private suspend fun <T> attempt(block: suspend () -> T): T? =
        try {
            block()
        } catch (error: CancellationException) {
            // A cancelled coroutine is an IllegalStateException too, and is not a failure.
            throw error
        } catch (error: IllegalStateException) {
            failed(error)
        } catch (error: IllegalArgumentException) {
            failed(error)
        } catch (error: android.database.SQLException) {
            failed(error)
        }

    private fun <T> failed(error: Exception): T? {
        Log.w(TAG, "household change failed", error)
        return null
    }

    private companion object {
        const val TAG = "PalliHousehold"
    }
}
