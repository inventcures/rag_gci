package org.inventcures.pallisahayak.app.ui

import android.Manifest
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.StringRes
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.material3.TimeInput
import androidx.compose.material3.rememberTimePickerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import org.inventcures.pallisahayak.R
import org.inventcures.pallisahayak.data.household.HouseholdNotice
import org.inventcures.pallisahayak.data.household.HouseholdUiState
import org.inventcures.pallisahayak.data.household.Severity
import org.inventcures.pallisahayak.data.household.Symptom
import org.inventcures.pallisahayak.data.local.HouseholdMemberEntity
import org.inventcures.pallisahayak.data.reminders.DailyReminder
import java.util.Locale

/**
 * The household screen: the people a participant cares for, a symptom noticed
 * about one of them, and their medicine reminders.
 *
 * Everything is chosen rather than typed, except a person's name and a medicine's
 * name. A name stays on the phone. A medicine's name is checked for an amount
 * before it is kept, because a reminder never carries a dose.
 *
 * One form is open at a time, under the person it is about, so the screen stays
 * short on a small phone and nobody records a symptom against the wrong person.
 */
@Composable
fun HouseholdScreen(
    state: HouseholdUiState,
    onAddMember: (String) -> Unit,
    onRecordObservation: (memberId: String, symptom: Symptom, severity: Severity) -> Unit,
    /** The medicine as typed, the sentence the reminder will say, and minutes since midnight. */
    onAddReminder: (medicine: String, text: String, minutesOfDay: Int) -> Unit,
    onRemoveReminder: (medicationRef: String) -> Unit,
    modifier: Modifier = Modifier,
) {
    // Which person's form is open, and which of the two.
    var openMember by rememberSaveable { mutableStateOf<String?>(null) }
    var openForm by rememberSaveable { mutableStateOf(MemberForm.NONE) }

    Column(
        modifier = modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(
            text = stringResource(R.string.household_title),
            style = MaterialTheme.typography.headlineSmall,
        )
        Text(
            text = stringResource(R.string.household_explainer),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        state.notice?.let { NoticeCard(it) }

        if (state.loaded && state.members.isEmpty()) {
            Text(
                text = stringResource(R.string.household_empty),
                style = MaterialTheme.typography.bodyLarge,
            )
        }

        state.members.forEach { member ->
            MemberCard(
                member = member,
                form = if (openMember == member.id) openForm else MemberForm.NONE,
                onOpen = { form ->
                    openMember = member.id
                    openForm = form
                },
                onRecordObservation = { symptom, severity ->
                    onRecordObservation(member.id, symptom, severity)
                    openForm = MemberForm.NONE
                },
                onAddReminder = { medicine, text, minutes ->
                    onAddReminder(medicine, text, minutes)
                    openForm = MemberForm.NONE
                },
            )
        }

        AddPerson(onAddMember)

        HorizontalDivider()
        Text(
            text = stringResource(R.string.reminders_title),
            style = MaterialTheme.typography.headlineSmall,
        )
        if (state.loaded && state.reminders.isEmpty()) {
            Text(
                text = stringResource(R.string.reminders_empty),
                style = MaterialTheme.typography.bodyLarge,
            )
        }
        state.reminders.forEach { reminder ->
            ReminderRow(reminder, onRemove = { onRemoveReminder(reminder.medicationRef) })
        }
    }
}

/** Which form is open under a person. */
enum class MemberForm { NONE, OBSERVATION, REMINDER }

@Composable
private fun MemberCard(
    member: HouseholdMemberEntity,
    form: MemberForm,
    onOpen: (MemberForm) -> Unit,
    onRecordObservation: (Symptom, Severity) -> Unit,
    onAddReminder: (medicine: String, text: String, minutesOfDay: Int) -> Unit,
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = member.displayName,
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.SemiBold,
            )
            when (form) {
                MemberForm.NONE -> {
                    WideButton(R.string.observation_open) { onOpen(MemberForm.OBSERVATION) }
                    WideOutlinedButton(R.string.reminder_open) { onOpen(MemberForm.REMINDER) }
                }
                MemberForm.OBSERVATION -> ObservationForm(onRecordObservation)
                MemberForm.REMINDER -> ReminderForm(member.displayName, onAddReminder)
            }
        }
    }
}

@Composable
private fun ObservationForm(onSave: (Symptom, Severity) -> Unit) {
    var symptom by rememberSaveable { mutableStateOf<Symptom?>(null) }
    var severity by rememberSaveable { mutableStateOf<Severity?>(null) }

    Text(
        text = stringResource(R.string.observation_symptom_title),
        style = MaterialTheme.typography.titleMedium,
    )
    Symptom.entries.forEach { option ->
        Choice(option.labelRes, selected = symptom == option) { symptom = option }
    }
    Text(
        text = stringResource(R.string.observation_severity_title),
        style = MaterialTheme.typography.titleMedium,
    )
    Severity.entries.forEach { option ->
        Choice(option.labelRes, selected = severity == option) { severity = option }
    }
    val chosenSymptom = symptom
    val chosenSeverity = severity
    WideButton(
        label = R.string.observation_save,
        enabled = chosenSymptom != null && chosenSeverity != null,
    ) {
        if (chosenSymptom != null && chosenSeverity != null) onSave(chosenSymptom, chosenSeverity)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ReminderForm(
    memberName: String,
    onSave: (medicine: String, text: String, minutesOfDay: Int) -> Unit,
) {
    var medicine by rememberSaveable { mutableStateOf("") }
    val time = rememberTimePickerState(initialHour = 8, initialMinute = 0, is24Hour = true)

    // Android 13 and later show a notification only once the participant allows
    // it. Asked here, when they make a reminder, which is when the reason is plain.
    // The reminder is saved whatever they answer; refusing it is reported when the
    // reminder would have appeared, in the log, and the screen says why it matters.
    val askToNotify = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {}

    val trimmed = medicine.trim()
    val text = stringResource(R.string.reminder_text_format, trimmed, memberName)

    OutlinedTextField(
        value = medicine,
        onValueChange = { medicine = it },
        label = { Text(stringResource(R.string.reminder_medicine_label)) },
        singleLine = true,
        modifier = Modifier.fillMaxWidth(),
    )
    Text(
        text = stringResource(R.string.reminder_time_title),
        style = MaterialTheme.typography.titleMedium,
    )
    TimeInput(state = time)
    Text(
        text = stringResource(R.string.reminder_notifications_needed),
        style = MaterialTheme.typography.bodyMedium,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
    )
    WideButton(label = R.string.reminder_save, enabled = trimmed.isNotEmpty()) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            askToNotify.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
        onSave(trimmed, text, time.hour * MINUTES_IN_HOUR + time.minute)
    }
}

@Composable
private fun AddPerson(onAdd: (String) -> Unit) {
    var name by rememberSaveable { mutableStateOf("") }

    OutlinedTextField(
        value = name,
        onValueChange = { name = it },
        label = { Text(stringResource(R.string.household_name_label)) },
        singleLine = true,
        modifier = Modifier.fillMaxWidth(),
    )
    WideButton(label = R.string.household_add_person, enabled = name.isNotBlank()) {
        onAdd(name.trim())
        name = ""
    }
}

@Composable
private fun ReminderRow(reminder: DailyReminder, onRemove: () -> Unit) {
    val time = clockTime(reminder.minutesOfDay)
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = time,
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.SemiBold,
            )
            Text(text = reminder.text, style = MaterialTheme.typography.bodyLarge)
            WideOutlinedButton(R.string.reminder_remove, onClick = onRemove)
        }
    }
}

/**
 * What happened, by colour as well as by words: green for stored, amber for
 * "this was not kept". A failure that looked like the success would be believed.
 */
@Composable
private fun NoticeCard(notice: HouseholdNotice) {
    val stored = notice == HouseholdNotice.OBSERVATION_SAVED || notice == HouseholdNotice.REMINDER_SAVED
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = if (stored) Color(0xFFF1F8F2) else Color(0xFFFFF4E5),
        ),
    ) {
        Text(
            text = stringResource(notice.messageRes),
            modifier = Modifier.padding(12.dp),
            style = MaterialTheme.typography.bodyLarge,
        )
    }
}

/** One option of a pick-one list: the whole row is the target, not just the dot. */
@Composable
private fun Choice(@StringRes label: Int, selected: Boolean, onSelect: () -> Unit) {
    val text = stringResource(label)
    val state = stringResource(
        if (selected) R.string.language_state_selected else R.string.language_state_not_selected,
    )
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onSelect)
            .padding(vertical = 8.dp)
            .semantics {
                contentDescription = text
                stateDescription = state
            },
    ) {
        RadioButton(selected = selected, onClick = onSelect)
        Text(
            text = text,
            style = MaterialTheme.typography.bodyLarge,
            modifier = Modifier.padding(start = 12.dp),
        )
    }
}

@Composable
private fun WideButton(@StringRes label: Int, enabled: Boolean = true, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        enabled = enabled,
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 56.dp),
    ) {
        Text(stringResource(label), style = MaterialTheme.typography.titleMedium)
    }
}

@Composable
private fun WideOutlinedButton(@StringRes label: Int, onClick: () -> Unit) {
    OutlinedButton(
        onClick = onClick,
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 56.dp),
    ) {
        Text(stringResource(label), style = MaterialTheme.typography.titleMedium)
    }
}

/** A time of day as the 24-hour clock shows it, 08:30. Digits only, in any language. */
private fun clockTime(minutesOfDay: Int): String = String.format(
    Locale.ROOT,
    "%02d:%02d", // not user-facing: a number format, digits and a colon
    minutesOfDay / MINUTES_IN_HOUR,
    minutesOfDay % MINUTES_IN_HOUR,
)

private const val MINUTES_IN_HOUR = 60

@get:StringRes
private val Symptom.labelRes: Int
    get() = when (this) {
        Symptom.PAIN -> R.string.symptom_pain
        Symptom.BREATHLESSNESS -> R.string.symptom_breathlessness
        Symptom.NAUSEA -> R.string.symptom_nausea
        Symptom.CONSTIPATION -> R.string.symptom_constipation
    }

@get:StringRes
private val Severity.labelRes: Int
    get() = when (this) {
        Severity.NONE -> R.string.severity_none
        Severity.MILD -> R.string.severity_mild
        Severity.MODERATE -> R.string.severity_moderate
        Severity.SEVERE -> R.string.severity_severe
        Severity.VERY_SEVERE -> R.string.severity_very_severe
    }

@get:StringRes
private val HouseholdNotice.messageRes: Int
    get() = when (this) {
        HouseholdNotice.OBSERVATION_SAVED -> R.string.observation_saved
        HouseholdNotice.OBSERVATION_FAILED -> R.string.observation_failed
        HouseholdNotice.REMINDER_SAVED -> R.string.reminder_saved
        HouseholdNotice.REMINDER_HAS_DOSE -> R.string.reminder_has_dose
        HouseholdNotice.FAILED -> R.string.problem_unexpected
    }
