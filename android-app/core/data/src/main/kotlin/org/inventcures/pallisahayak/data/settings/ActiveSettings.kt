package org.inventcures.pallisahayak.data.settings

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.inventcures.pallisahayak.data.model.CareRole
import org.inventcures.pallisahayak.data.model.Language
import org.inventcures.pallisahayak.data.prefs.PreferencesStore
import org.inventcures.pallisahayak.data.repository.SessionContext

/**
 * The Active Role and the language, as the settings screen switches them.
 *
 * A phone is handed between an ASHA Worker, a family caregiver and a patient, so
 * the person holding it switches the Active Role from settings, with one tap and no
 * PIN (ADR 0006: the Care Role is an analysis label, not an access boundary). The
 * language is switched the same way. Both are remembered in [store] across launches.
 *
 * Every interaction is recorded under [applyTo]'s session: the registered
 * participant, with the Active Role and language in force when it was asked. Until
 * someone switches, that is the role and language the phone registered with.
 */
class ActiveSettings(private val store: PreferencesStore) {

    private val _role = MutableStateFlow(store.careRole())
    private val _language = MutableStateFlow(store.language())

    /** The Active Role switched to in settings, or null while it is the registered one. */
    val role: StateFlow<CareRole?> = _role.asStateFlow()

    /** The language chosen, or null before the first-run picker. */
    val language: StateFlow<Language?> = _language.asStateFlow()

    /** Switch the Active Role. The next interaction is recorded under it. */
    fun switchRole(role: CareRole) {
        store.setCareRole(role)
        _role.value = role
    }

    /** Switch the language. The next question is asked, and recorded, in it. */
    fun switchLanguage(language: Language) {
        store.setLanguage(language)
        _language.value = language
    }

    /** The Active Role for a registered [session]: the switched one, else the registered one. */
    fun activeRole(session: SessionContext): CareRole =
        role.value ?: CareRole.fromWire(session.careRole)

    /**
     * The session the next interaction runs under.
     *
     * The registered participant, site and release, with the Active Role and the
     * language in force now. A registered language in an older spelling ("od-IN")
     * is sent in the table's ("or-IN").
     */
    fun applyTo(session: SessionContext): SessionContext = session.copy(
        careRole = role.value?.wireName ?: session.careRole,
        language = (language.value ?: Language.fromTag(session.language))?.tag ?: session.language,
    )
}
