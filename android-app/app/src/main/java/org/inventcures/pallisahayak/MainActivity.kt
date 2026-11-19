package org.inventcures.pallisahayak

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import org.inventcures.pallisahayak.data.ServiceLocator
import org.inventcures.pallisahayak.reminders.ReminderAlarms

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Built before setContent, because the composition reads the repository.
        // The alarm for the next medicine reminder belongs to this module; the data
        // layer only says when the reminders have changed. Set again on every open
        // as well, in case the phone dropped it (a force stop clears alarms).
        ServiceLocator.onRemindersChanged = { ReminderAlarms.schedule(applicationContext) }
        ServiceLocator.initialise(applicationContext)
        ReminderAlarms.schedule(applicationContext)
        setContent { AppRoot() }
    }
}
