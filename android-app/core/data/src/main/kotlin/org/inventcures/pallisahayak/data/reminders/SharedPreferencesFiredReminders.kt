package org.inventcures.pallisahayak.data.reminders

import android.content.Context
import android.content.SharedPreferences

/**
 * [FiredReminders] kept in preferences: for each reminder, the last day it fired.
 *
 * Preferences rather than a table. It is one number a reminder, it is not study
 * data, and a column for it would be a database migration on every phone.
 */
class SharedPreferencesFiredReminders(context: Context) : FiredReminders {

    private val prefs: SharedPreferences =
        context.applicationContext.getSharedPreferences(NAME, Context.MODE_PRIVATE)

    override fun wasFired(reminderId: String, epochDay: Long): Boolean =
        prefs.getLong(reminderId, NEVER) == epochDay

    override fun markFired(reminderId: String, epochDay: Long) {
        prefs.edit().putLong(reminderId, epochDay).apply()
    }

    private companion object {
        const val NAME = "palli_fired_reminders"
        const val NEVER = Long.MIN_VALUE
    }
}
