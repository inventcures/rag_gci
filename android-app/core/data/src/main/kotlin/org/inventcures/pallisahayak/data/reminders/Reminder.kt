package org.inventcures.pallisahayak.data.reminders

import androidx.room.ColumnInfo
import androidx.room.Dao
import androidx.room.Entity
import androidx.room.PrimaryKey
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

/**
 * One medication reminder that lives on the device.
 *
 * Criterion 5: reminders already on the device keep firing offline. That means the
 * schedule and the text are both stored here, so firing them needs no server and no
 * network. A reminder that had to ask the server what to say would stop working at
 * exactly the moment a worker needs it, which is the moment they are in a village
 * with no signal.
 *
 * The text is stored rather than fetched because the whole point is that it is
 * available without a connection. It was written when the reminder was created, and
 * it does not change between then and the dose being taken.
 */
@Entity(tableName = "reminders")
data class ReminderEntity(
    @PrimaryKey
    val id: String,

    /** Server-side medication reference, so a sync can reconcile it later. */
    @ColumnInfo(name = "medication_ref")
    val medicationRef: String,

    /**
     * What is said when the reminder fires.
     *
     * Plain care language. It never contains a dose: the boundary that applies to an
     * answer applies to a reminder too, and a spoken reminder is the one thing a
     * worker cannot easily double-check before a family acts on it.
     */
    @ColumnInfo(name = "reminder_text")
    val reminderText: String,

    /** Minutes since local midnight. 8:30 is 510. */
    @ColumnInfo(name = "minutes_of_day")
    val minutesOfDay: Int,

    /** 0 = Sunday. Kept because a family schedule is weekly. */
    @ColumnInfo(name = "day_of_week")
    val dayOfWeek: Int,

    /** Which language to speak it in, so it matches the rest of the session. */
    @ColumnInfo(name = "language_tag")
    val languageTag: String,

    @ColumnInfo(name = "enabled")
    val enabled: Boolean,

    @ColumnInfo(name = "created_at")
    val createdAt: Long,
)

@Dao
interface ReminderDao {

    @androidx.room.Upsert
    suspend fun upsert(reminder: ReminderEntity)

    @androidx.room.Upsert
    suspend fun upsertAll(reminders: List<ReminderEntity>)

    @Query("SELECT * FROM reminders WHERE enabled = 1 ORDER BY minutes_of_day ASC")
    fun observeEnabled(): Flow<List<ReminderEntity>>

    @Query("SELECT * FROM reminders WHERE enabled = 1 ORDER BY minutes_of_day ASC")
    suspend fun enabled(): List<ReminderEntity>

    @Query("SELECT * FROM reminders WHERE id = :id")
    suspend fun find(id: String): ReminderEntity?

    @Query("UPDATE reminders SET enabled = :enabled WHERE id = :id")
    suspend fun setEnabled(id: String, enabled: Boolean)

    @Query("DELETE FROM reminders WHERE id = :id")
    suspend fun delete(id: String)
}

/** One reminder that should fire now. */
data class DueReminder(
    val id: String,
    val medicationRef: String,
    val text: String,
    val languageTag: String,
)

/**
 * Decides which reminders are due, with no network involved.
 *
 * Pure on purpose. The logic that determines whether a reminder fires is the part
 * that has to be right at a site with no signal, and a pure function is the only way
 * to be sure of it without a device and a village. Everything around it, the alarm
 * that wakes the process and the audio that plays, is already-computed inputs to it.
 */
class ReminderDueResolver {

    /**
     * The reminders that should fire at [minutesNow] on [dayOfWeek].
     *
     * A window rather than an exact minute, because an alarm can be delivered late
     * when a device has been asleep or a battery saver has deferred it. Matching the
     * exact minute would silently drop a reminder that had been delayed, which is
     * the failure that matters: the family does not know a dose was missed.
     */
    fun due(
        reminders: List<ReminderEntity>,
        minutesNow: Int,
        dayOfWeek: Int,
        windowMinutes: Int = 30,
    ): List<DueReminder> =
        reminders
            .filter { it.enabled && it.dayOfWeek == dayOfWeek }
            .filter { reminder ->
                val delta = minutesNow - reminder.minutesOfDay
                delta in 0..windowMinutes
            }
            .sortedBy { it.minutesOfDay }
            .map {
                DueReminder(
                    id = it.id,
                    medicationRef = it.medicationRef,
                    text = it.reminderText,
                    languageTag = it.languageTag,
                )
            }
}