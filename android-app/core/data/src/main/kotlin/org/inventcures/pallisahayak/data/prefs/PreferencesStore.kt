package org.inventcures.pallisahayak.data.prefs

import android.content.Context
import android.content.SharedPreferences
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.inventcures.pallisahayak.data.model.CareRole
import org.inventcures.pallisahayak.data.model.Language

/**
 * What the app remembers between launches.
 *
 * Small on purpose. The language is chosen once and never re-asked, because a
 * worker who is asked on every launch stops trusting the setting. The Care Role
 * changes often, since a phone is handed back and forth, and switching it must not
 * feel like an account change.
 *
 * An interface rather than a concrete store so the tests do not need an Android
 * context. The alternative, reaching for SharedPreferences inside a ViewModel,
 * makes the persistence untestable and therefore unchecked.
 */
interface PreferencesStore {
    /** Null until a language has been chosen. Never a default. */
    fun language(): Language?

    /** True once the worker has made the choice, so onboarding does not reappear. */
    fun languageChosen(): Boolean

    fun careRole(): CareRole

    fun setLanguage(language: Language)

    fun setCareRole(role: CareRole)

    fun clear()
}

/**
 * The real store, backed by SharedPreferences.
 *
 * The language is stored as its tag rather than as an enum ordinal, because an
 * ordinal changes meaning the moment someone reorders the enum, and this value
 * outlives the app version. Reading an unrecognised tag yields null, which
 * re-asks the question rather than silently answering in the wrong language.
 */
class SharedPreferencesStore(context: Context) : PreferencesStore {

    private val prefs: SharedPreferences =
        context.applicationContext.getSharedPreferences(NAME, Context.MODE_PRIVATE)

    override fun language(): Language? = Language.fromTag(prefs.getString(KEY_LANGUAGE, null))

    override fun languageChosen(): Boolean = prefs.contains(KEY_LANGUAGE)

    override fun careRole(): CareRole = CareRole.fromWire(prefs.getString(KEY_ROLE, null))

    override fun setLanguage(language: Language) {
        prefs.edit().putString(KEY_LANGUAGE, language.tag).apply()
    }

    override fun setCareRole(role: CareRole) {
        // No PIN, no confirmation, no lock. One tap, per ADR 0006.
        prefs.edit().putString(KEY_ROLE, role.wireName).apply()
    }

    override fun clear() {
        prefs.edit().clear().apply()
    }

    private companion object {
        const val NAME = "palli-sahayak-prefs"
        const val KEY_LANGUAGE = "language_tag"
        const val KEY_ROLE = "care_role"
    }
}

/**
 * An in-memory store for tests and for previews.
 *
 * Exposed rather than private so tests can assert on it directly instead of
 * inferring state from behaviour.
 */
class InMemoryPreferencesStore(
    private var language: Language? = null,
    private var role: CareRole = CareRole.ASHA_WORKER,
) : PreferencesStore {

    private val changes = MutableStateFlow(0)

    /** Increments on every write, so a test can wait for persistence to happen. */
    val changeCount: StateFlow<Int> = changes.asStateFlow()

    override fun language(): Language? = language

    override fun languageChosen(): Boolean = language != null

    override fun careRole(): CareRole = role

    override fun setLanguage(language: Language) {
        this.language = language
        changes.value += 1
    }

    override fun setCareRole(role: CareRole) {
        this.role = role
        changes.value += 1
    }

    override fun clear() {
        language = null
        role = CareRole.ASHA_WORKER
        changes.value += 1
    }
}