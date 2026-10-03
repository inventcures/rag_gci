package org.inventcures.pallisahayak.data.local

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

/**
 * The app's on-device store.
 *
 * Two tables only. Interactions are written by the server-side recording path and
 * pushed when a connection exists; cached answers come from the offline bundle
 * built by `offline/cache_builder.py`.
 *
 * The database is deliberately small. The app holds no clinical records of its
 * own: patient observations live on the server and sync through the mobile API, so
 * a lost or stolen device does not expose a patient history.
 */
@Database(
    entities = [InteractionEntity::class, CachedAnswerEntity::class],
    version = 1,
    exportSchema = true,
)
abstract class PalliDatabase : RoomDatabase() {

    abstract fun interactions(): InteractionDao

    abstract fun cachedAnswers(): CachedAnswerDao

    companion object {
        private const val NAME = "palli-sahayak.db"

        @Volatile
        private var instance: PalliDatabase? = null

        /**
         * One database per process.
         *
         * Opening a SQLite file twice in the same process is a real source of
         * corruption, and two `PalliDatabase` objects would each hold their own
         * connection. `synchronized` on the first caller, `@Volatile` so later
         * readers on other threads see the assignment.
         */
        fun get(context: Context): PalliDatabase =
            instance ?: synchronized(this) {
                instance ?: build(context.applicationContext).also { instance = it }
            }

        private fun build(context: Context): PalliDatabase =
            // Deliberately no fallbackToDestructiveMigration.
            //
            // Version 1 has not shipped, so nothing needs migrating yet. But if a
            // later version adds a column and this builder silently drops the
            // tables on upgrade, it deletes the recorded interactions that the
            // study protocol depends on, and the loss is invisible until someone
            // counts rows six weeks later. Without a fallback, a version mismatch
            // throws at open time instead, which is a loud failure in front of the
            // developer who caused it. That is the right trade for study data.
            Room.databaseBuilder(context, PalliDatabase::class.java, NAME)
                .build()

        /** Test seam. A test cannot share the process-wide instance above. */
        internal fun resetForTests() {
            instance?.close()
            instance = null
        }
    }
}
