# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reactor configuration registry

"""Extension-safe fusion-reactor configuration registry."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .vocabulary import (
    ConfinementFamily,
    require_identifier,
    require_semver,
    require_sha256,
    require_text,
)

REACTOR_REGISTRY_VERSION = "1.1.0"
REACTOR_REGISTRY_V1_0_0_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class ReactorConfiguration:
    """One reactor configuration known to a registry.

    Parameters
    ----------
    identifier : str
        Stable configuration identifier. Extensions use a namespaced identifier
        such as "institution.example:novel_configuration".
    confinement_family : ConfinementFamily
        Broad physical family.
    topology : str
        Human-readable geometry or topology description.
    """

    identifier: str
    confinement_family: ConfinementFamily
    topology: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "identifier",
            require_identifier(self.identifier, field="configuration identifier"),
        )
        object.__setattr__(
            self, "topology", require_text(self.topology, field="topology")
        )

    def to_record(self) -> dict[str, str]:
        """Return a deterministic JSON-compatible record.

        Returns
        -------
        dict[str, str]
            Configuration identifier, family, and topology fields.
        """
        return {
            "confinement_family": self.confinement_family.value,
            "identifier": self.identifier,
            "topology": self.topology,
        }


@dataclass(frozen=True, slots=True)
class ReactorConfigurationRegistry:
    """Immutable registry of reactor configurations and aliases."""

    version: str
    configurations: Mapping[str, ReactorConfiguration]
    aliases: Mapping[str, str]

    def __post_init__(self) -> None:
        version = require_semver(self.version, field="registry version")
        canonical = dict(self.configurations)
        aliases = dict(self.aliases)
        if not canonical:
            raise ValueError("reactor registry requires at least one configuration")
        for identifier, configuration in canonical.items():
            if identifier != configuration.identifier:
                raise ValueError("registry key must equal configuration identifier")
        for alias, target in aliases.items():
            require_identifier(alias, field="configuration alias")
            if target not in canonical:
                raise ValueError(
                    f"alias {alias!r} targets unknown configuration {target!r}"
                )
            if alias in canonical:
                raise ValueError(f"alias {alias!r} collides with a configuration")
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "configurations", MappingProxyType(canonical))
        object.__setattr__(self, "aliases", MappingProxyType(aliases))

    def resolve(self, identifier: str) -> ReactorConfiguration:
        """Resolve a canonical identifier or alias.

        Parameters
        ----------
        identifier : str
            Canonical reactor configuration identifier or registered alias.

        Returns
        -------
        ReactorConfiguration
            Registered canonical configuration.

        Raises
        ------
        ValueError
            If the identifier is not registered.
        """
        key = require_identifier(identifier, field="configuration")
        canonical = self.aliases.get(key, key)
        try:
            return self.configurations[canonical]
        except KeyError as exc:
            raise ValueError(f"unregistered reactor configuration: {key}") from exc

    def register(
        self,
        configuration: ReactorConfiguration,
        *,
        aliases: tuple[str, ...] = (),
    ) -> ReactorConfigurationRegistry:
        """Return a new registry containing one namespaced extension.

        Built-in identifiers cannot be shadowed. Extensions must contain a
        namespace separator so local names cannot silently acquire new meaning.

        Parameters
        ----------
        configuration : ReactorConfiguration
            Namespaced configuration to add.
        aliases : tuple[str, ...]
            Optional namespaced aliases for the new configuration.

        Returns
        -------
        ReactorConfigurationRegistry
            New immutable registry containing the extension.

        Raises
        ------
        ValueError
            If identifiers are not namespaced, collide, or are invalid.
        """
        if ":" not in configuration.identifier:
            raise ValueError("extension configuration must use a namespaced identifier")
        if configuration.identifier in self.configurations:
            raise ValueError(
                f"reactor configuration already registered: {configuration.identifier}"
            )
        updated = dict(self.configurations)
        updated[configuration.identifier] = configuration
        updated_aliases = dict(self.aliases)
        for alias in aliases:
            canonical_alias = require_identifier(alias, field="configuration alias")
            if canonical_alias in updated or canonical_alias in updated_aliases:
                raise ValueError(
                    f"configuration alias already registered: {canonical_alias}"
                )
            if ":" not in canonical_alias:
                raise ValueError("extension alias must use a namespaced identifier")
            updated_aliases[canonical_alias] = configuration.identifier
        return ReactorConfigurationRegistry(
            version=self.version,
            configurations=updated,
            aliases=updated_aliases,
        )

    def to_record(self) -> dict[str, object]:
        """Return the sorted deterministic registry record.

        Returns
        -------
        dict[str, object]
            JSON-compatible registry version, aliases, and configurations.
        """
        return {
            "aliases": dict(sorted(self.aliases.items())),
            "configurations": [
                self.configurations[key].to_record()
                for key in sorted(self.configurations)
            ],
            "version": self.version,
        }

    @property
    def digest(self) -> str:
        """Return the SHA-256 digest of the canonical registry record."""
        payload = json.dumps(
            self.to_record(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _configuration(
    identifier: str,
    family: ConfinementFamily,
    topology: str,
) -> ReactorConfiguration:
    """Build one built-in configuration."""
    return ReactorConfiguration(identifier, family, topology)


_BUILTIN_CONFIGURATIONS = (
    _configuration(
        "conventional_tokamak", ConfinementFamily.MAGNETIC_CLOSED, "axisymmetric torus"
    ),
    _configuration(
        "spherical_tokamak",
        ConfinementFamily.MAGNETIC_CLOSED,
        "low-aspect-ratio axisymmetric torus",
    ),
    _configuration(
        "stellarator", ConfinementFamily.MAGNETIC_CLOSED, "three-dimensional torus"
    ),
    _configuration(
        "heliotron", ConfinementFamily.MAGNETIC_CLOSED, "helical-coil torus"
    ),
    _configuration(
        "torsatron", ConfinementFamily.MAGNETIC_CLOSED, "continuous-helical-coil torus"
    ),
    _configuration(
        "reversed_field_pinch",
        ConfinementFamily.MAGNETIC_CLOSED,
        "relaxed-current torus",
    ),
    _configuration("spheromak", ConfinementFamily.MAGNETIC_CLOSED, "compact toroid"),
    _configuration(
        "field_reversed_configuration",
        ConfinementFamily.MAGNETIC_CLOSED,
        "field-reversed compact toroid",
    ),
    _configuration(
        "simple_magnetic_mirror",
        ConfinementFamily.MAGNETIC_OPEN,
        "linear magnetic mirror",
    ),
    _configuration(
        "tandem_mirror", ConfinementFamily.MAGNETIC_OPEN, "multi-cell linear mirror"
    ),
    _configuration(
        "gas_dynamic_mirror",
        ConfinementFamily.MAGNETIC_OPEN,
        "collisional linear mirror",
    ),
    _configuration("cusp", ConfinementFamily.MAGNETIC_OPEN, "magnetic cusp"),
    _configuration(
        "polywell",
        ConfinementFamily.MAGNETIC_OPEN,
        "electrostatically biased magnetic cusp",
    ),
    _configuration(
        "levitated_dipole", ConfinementFamily.MAGNETIC_OPEN, "closed dipole field"
    ),
    _configuration("z_pinch", ConfinementFamily.SELF_MAGNETIC, "axial-current pinch"),
    _configuration(
        "sheared_flow_z_pinch",
        ConfinementFamily.SELF_MAGNETIC,
        "flow-stabilized axial-current pinch",
    ),
    _configuration(
        "theta_pinch", ConfinementFamily.SELF_MAGNETIC, "azimuthal-current pinch"
    ),
    _configuration(
        "dense_plasma_focus",
        ConfinementFamily.SELF_MAGNETIC,
        "coaxial plasma-focus pinch",
    ),
    _configuration(
        "laser_icf_direct_drive",
        ConfinementFamily.INERTIAL,
        "spherical direct-drive target",
    ),
    _configuration(
        "laser_icf_indirect_drive",
        ConfinementFamily.INERTIAL,
        "hohlraum indirect-drive target",
    ),
    _configuration(
        "laser_icf_fast_or_shock_ignition",
        ConfinementFamily.INERTIAL,
        "staged laser-driven target",
    ),
    _configuration(
        "ion_beam_icf", ConfinementFamily.INERTIAL, "particle-beam-driven target"
    ),
    _configuration(
        "pulsed_electron_beam_icf",
        ConfinementFamily.INERTIAL,
        "electron-beam-driven target",
    ),
    _configuration(
        "projectile_or_impact_icf", ConfinementFamily.INERTIAL, "impact-driven target"
    ),
    _configuration(
        "maglif", ConfinementFamily.MAGNETO_INERTIAL, "magnetized cylindrical liner"
    ),
    _configuration(
        "plasma_jet_mif",
        ConfinementFamily.MAGNETO_INERTIAL,
        "converging plasma-jet liner",
    ),
    _configuration(
        "mechanical_or_liquid_liner_mif",
        ConfinementFamily.MAGNETO_INERTIAL,
        "material-liner compression",
    ),
    _configuration(
        "frc_compression_mif",
        ConfinementFamily.MAGNETO_INERTIAL,
        "compressed field-reversed configuration",
    ),
    _configuration(
        "gridded_iec",
        ConfinementFamily.ELECTROSTATIC,
        "gridded electrostatic potential well",
    ),
    _configuration(
        "colliding_beam",
        ConfinementFamily.BEAM_TARGET,
        "counter-propagating particle beams",
    ),
    _configuration(
        "beam_target", ConfinementFamily.BEAM_TARGET, "energetic beam on target"
    ),
    _configuration(
        "fusion_fission_hybrid",
        ConfinementFamily.HYBRID,
        "fusion source with subcritical blanket",
    ),
)

_EXTENSION_CONFIGURATIONS = (
    _configuration(
        "scpn.reactor_systems:lattice_confinement_fusion",
        ConfinementFamily.EXTENSION,
        "deuterated metal lattice under external driver",
    ),
    _configuration(
        "scpn.reactor_systems:muon_catalysed_fusion",
        ConfinementFamily.EXTENSION,
        "muon-catalysed hydrogen-isotope target",
    ),
)

_REGISTRY_ALIASES = {
    "frc": "field_reversed_configuration",
    "iec": "gridded_iec",
    "mif_maglif": "maglif",
    "rfx": "reversed_field_pinch",
}

REACTOR_REGISTRY_V1_0_0 = ReactorConfigurationRegistry(
    version=REACTOR_REGISTRY_V1_0_0_VERSION,
    configurations={
        configuration.identifier: configuration
        for configuration in _BUILTIN_CONFIGURATIONS
    },
    aliases=_REGISTRY_ALIASES,
)

DEFAULT_REACTOR_REGISTRY = ReactorConfigurationRegistry(
    version=REACTOR_REGISTRY_VERSION,
    configurations={
        configuration.identifier: configuration
        for configuration in (*_BUILTIN_CONFIGURATIONS, *_EXTENSION_CONFIGURATIONS)
    },
    aliases=_REGISTRY_ALIASES,
)

_REACTOR_REGISTRY_RELEASES = MappingProxyType(
    {
        (REACTOR_REGISTRY_V1_0_0.version, REACTOR_REGISTRY_V1_0_0.digest): (
            REACTOR_REGISTRY_V1_0_0
        ),
        (DEFAULT_REACTOR_REGISTRY.version, DEFAULT_REACTOR_REGISTRY.digest): (
            DEFAULT_REACTOR_REGISTRY
        ),
    }
)


def resolve_reactor_registry_release(
    version: str,
    digest: str,
) -> ReactorConfigurationRegistry:
    """Resolve one exact immutable registry release.

    Producer objects remain bound to the release under which their bytes were
    authored. Recognising that release preserves custody; it does not silently
    upgrade or reinterpret the producer object against the current registry.

    Parameters
    ----------
    version : str
        Exact semantic version declared by the producer.
    digest : str
        Exact SHA-256 digest of the canonical registry record.

    Returns
    -------
    ReactorConfigurationRegistry
        Recognised immutable release.

    Raises
    ------
    ValueError
        If the version and digest pair is not a recognised release.
    """
    key = (
        require_semver(version, field="registry version"),
        require_sha256(digest, field="registry digest"),
    )
    try:
        return _REACTOR_REGISTRY_RELEASES[key]
    except KeyError as exc:
        raise ValueError("unrecognised reactor registry release") from exc


__all__ = [
    "DEFAULT_REACTOR_REGISTRY",
    "REACTOR_REGISTRY_VERSION",
    "REACTOR_REGISTRY_V1_0_0",
    "REACTOR_REGISTRY_V1_0_0_VERSION",
    "ReactorConfiguration",
    "ReactorConfigurationRegistry",
    "resolve_reactor_registry_release",
]
