"""Michel reconstruction module."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numexpr as ne
import numpy as np

from spine.calib.lifetime import LifetimeCalibrator
from spine.constants import MICHL_SHP, TRACK_SHP
from spine.geo import GeoManager
from spine.math.distance import closest_pair
from spine.post.base import PostBase

__all__ = [
    "MichelDistanceProcessor",
    "MichelAngleProcessor",
    "MichelIonizationProcessor",
    "MichelCCEnergyProcessor",
]


class MichelDistanceProcessor(PostBase):
    """Pairs Michel electrons with a candidate parent muon and stores the
    shortest distance between the two.

    An event may contain several Michel-muon pairs (e.g. multiple stopping
    muons in the same image). Pairing is done independently within each
    interaction: every Michel electron in an interaction is matched to the
    closest track-like particle in that same interaction, and the resulting
    distance is stored on the Michel particle.

    Note: if the `cathode_crosser` post-processor is used with
    `merge_crossers=True`, it must run *before* this processor. Merging
    renumbers particle IDs, and this processor stores `muon_match_id` by ID
    -- if it ran first, the stored ID would go stale as soon as merging
    renumbers the particle list.
    """

    name = "michel_muon_distance"

    def __init__(
        self,
        run_mode: str = "reco",
        truth_point_mode: str = "points",
    ) -> None:
        """Store the Michel distance processor parameters.

        Parameters
        ----------
        run_mode : str, default "reco"
            Whether to run this processor on reconstructed ('reco'), true
            ('truth') or both ('both'/'all') interactions
        truth_point_mode : str, default "points"
            Attribute used to fetch the point coordinates of true particles
        """
        super().__init__(
            "interaction",
            run_mode=run_mode,
            truth_point_mode=truth_point_mode,
        )

    def process(self, data: Mapping[str, Any]) -> None:
        """Compute the Michel-to-muon distance for each interaction in one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        for k in self.interaction_keys:
            for inter in data[k]:
                michels = [p for p in inter.particles if p.shape == MICHL_SHP]
                muons = [p for p in inter.particles if p.shape == TRACK_SHP]

                if not michels or not muons:
                    continue

                for michel in michels:
                    michel_points = self.get_points(michel)

                    best_dist, best_muon = np.inf, None
                    for muon in muons:
                        muon_points = self.get_points(muon)
                        _, _, dist = closest_pair(michel_points, muon_points)
                        if dist < best_dist:
                            best_dist, best_muon = dist, muon

                    michel.michel_muon_distance = float(best_dist)
                    michel.muon_match_id = best_muon.id


class MichelAngleProcessor(PostBase):
    """Computes the angle between a Michel electron and its paired muon.

    Uses the Michel start direction and the muon end direction as a
    first approximation of the decay kink angle. Relies on `muon_match_id`,
    which is set by :class:`MichelDistanceProcessor`.
    """

    name = "michel_muon_angle"

    _upstream = ("direction", "michel_muon_distance")

    def __init__(
        self,
        run_mode: str = "reco",
        truth_point_mode: str = "points",
    ) -> None:
        """Store the Michel angle processor parameters.

        Parameters
        ----------
        run_mode : str, default "reco"
            Whether to run this processor on reconstructed ('reco'), true
            ('truth') or both ('both'/'all') interactions
        truth_point_mode : str, default "points"
            Attribute used to fetch the point coordinates of true particles
        """
        super().__init__(
            "interaction",
            run_mode=run_mode,
            truth_point_mode=truth_point_mode,
        )

    def process(self, data: Mapping[str, Any]) -> None:
        """Compute the Michel-to-muon angle for each interaction in one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        for k in self.interaction_keys:
            for inter in data[k]:
                particles_by_id = {p.id: p for p in inter.particles}

                for part in inter.particles:
                    if part.shape != MICHL_SHP or part.muon_match_id == -1:
                        continue

                    muon = particles_by_id.get(part.muon_match_id)
                    if muon is None:
                        continue

                    # Angle in radians
                    michel_dir = part.start_dir / np.linalg.norm(part.start_dir)
                    muon_dir = muon.end_dir / np.linalg.norm(muon.end_dir)
                    cosang = np.clip(np.dot(michel_dir, muon_dir), -1.0, 1.0)

                    part.michel_muon_angle = float(np.arccos(cosang))


class MichelIonizationProcessor(PostBase):
    """Isolates and stores the primary ionization deposits of a Michel electron.

    The primary ionization is
    restricted to the subset of the Michel's own points that come from
    fragments which were themselves classified as Michel-shaped.

    Runs after `cathode_crosser` (merging/position adjustment) and
    `michel_cathode_energy` (lifetime recorrection) so that, for a Michel
    matched to a cathode-crossing muon, the primary ionization -- and the
    `primary_calo_ke` derived from it -- is drawn from the corrected
    depositions rather than the stale, pre-merge ones.
    """

    name = "michel_ionization"

    _upstream = ("cathode_crosser", "michel_cc_energy")

    def __init__(
        self,
        scaling: float | str = 1.0,
        shower_fudge: float | str = 1.0,
        run_mode: str = "reco",
        truth_point_mode: str = "points",
        truth_dep_mode: str = "depositions",
    ) -> None:
        """Store the Michel ionization processor parameters.

        Parameters
        ----------
        scaling : float or str, default 1.
            Global ADC-to-MeV conversion factor, same
            convention as the `calo_ke` post-processor
        shower_fudge : float or str, default 1.
            Shower energy fudge factor, same
            convention as the `calo_ke` post-processor.
        run_mode : str, default "reco"
            Whether to run this processor on reconstructed ('reco'), true
            ('truth') or both ('both'/'all') particles
        truth_point_mode : str, default "points"
            Attribute used to fetch the point coordinates of true particles
        truth_dep_mode : str, default "depositions"
            Attribute used to fetch the depositions of true particles
        """
        super().__init__(
            "particle",
            run_mode=run_mode,
            truth_point_mode=truth_point_mode,
            truth_dep_mode=truth_dep_mode,
        )

        scaling = float(ne.evaluate(scaling)) if isinstance(scaling, str) else scaling
        shower_fudge = (
            float(ne.evaluate(shower_fudge))
            if isinstance(shower_fudge, str)
            else shower_fudge
        )
        self.scaling = scaling * shower_fudge

    def process(self, data: Mapping[str, Any]) -> None:
        """Isolate the primary ionization deposits of Michel electrons in one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        for k in self.particle_keys:
            for part in data[k]:
                if part.shape != MICHL_SHP:
                    continue

                # Gather the indexes of every fragment that was itself
                # classified as Michel-shaped
                frag_index = [
                    self.get_index(frag)
                    for frag in part.fragments
                    if frag.shape == MICHL_SHP
                ]
                if not frag_index:
                    part.primary_depositions = np.empty(0, dtype=np.float32)
                    part.primary_calo_ke = 0.0
                    continue

                frag_index = np.concatenate(frag_index)

                # Restrict the particle's own depositions to that subset.
                # Prefer the cathode-corrected depositions, if available
                primary_mask = np.isin(self.get_index(part), frag_index)
                if len(part.corrected_depositions) > 0:
                    depositions = part.corrected_depositions
                else:
                    depositions = self.get_depositions(part)

                part.primary_depositions = depositions[primary_mask].astype(
                    np.float32
                )
                part.primary_calo_ke = self.scaling * float(
                    np.sum(part.primary_depositions)
                )


class MichelCCEnergyProcessor(PostBase):
    """Recomputes the electron lifetime correction for a Michel electron
    matched to a cathode-crossing parent muon.

    The `cathode_crosser` post-processor merges a muon track that was split
    at the cathode and shifts every particle in the same interaction by the same
    offset along the drift direction, so that everything lines up spatially.
    That shift changes the true drift distance.

    This processor undoes that stale, wrong-position correction and
    reapplies the lifetime correction at the shifted position, for the
    Michel electron only. It does not touch gain or recombination, since
    neither depends on the drift-axis position.

    Requires `cathode_crosser` (to shift the points and set
    `is_cathode_crosser`/`cathode_offset` on the muon) and
    `michel_muon_distance` (to set `muon_match_id` on the Michel) to have
    already run.
    """

    name = "michel_cc_energy"

    _upstream = ("cathode_crosser", "michel_muon_distance")

    # Set of data keys needed for this post-processor to operate
    _keys = (("run_info", False),)

    def __init__(
        self,
        lifetime: float | list[float] | None = None,
        driftv: float | list[float] | None = None,
        lifetime_db: str | dict | None = None,
        driftv_db: str | dict | None = None,
        undo_original_calib: bool = True,
        run_mode: str = "reco",
        truth_point_mode: str = "points",
        truth_dep_mode: str = "depositions",
    ) -> None:
        """Store the Michel cathode energy processor parameters.

        Parameters
        ----------
        lifetime : Union[float, List[float]], optional
            Electron lifetime in microseconds, see `LifetimeCalibrator`
        driftv : Union[float, List[float]], optional
            Electron drift velocity in cm/us, see `LifetimeCalibrator`
        lifetime_db : Union[str, dict], optional
            Lifetime database, see `LifetimeCalibrator`
        driftv_db : Union[str, dict], optional
            Drift velocity database, see `LifetimeCalibrator`
        undo_original_calib : bool, default True
            If True, first undoes the lifetime correction already applied
            to the Michel's depositions at its pre-shift position, before
            reapplying it at the shifted position. If the depositions were
            never lifetime-corrected in the first place, set to False.
        run_mode : str, default "reco"
            Whether to run this processor on reconstructed ('reco'), true
            ('truth') or both ('both'/'all') interactions
        truth_point_mode : str, default "points"
            Attribute used to fetch the point coordinates of true particles
        truth_dep_mode : str, default "depositions"
            Attribute used to fetch the depositions of true particles
        """
        super().__init__(
            "interaction",
            run_mode=run_mode,
            truth_point_mode=truth_point_mode,
            truth_dep_mode=truth_dep_mode,
        )

        # Fetch the detector geometry instance, build the lifetime calibrator
        self.geo = GeoManager.get_instance()
        self.calibrator = LifetimeCalibrator(
            num_tpcs=self.geo.tpc.num_chambers,
            lifetime=lifetime,
            driftv=driftv,
            lifetime_db=lifetime_db,
            driftv_db=driftv_db,
        )
        self.undo_original_calib = undo_original_calib
        self.scaling = scaling * shower_fudge

    def process(self, data: Mapping[str, Any]) -> None:
        """Recompute the cathode-corrected energy of Michel electrons paired
        to a cathode-crossing muon, in one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        run_id = None
        if "run_info" in data:
            run_id = data["run_info"].run

        for k in self.interaction_keys:
            for inter in data[k]:
                particles_by_id = {p.id: p for p in inter.particles}

                # Loop over the Michel electrons paired to a cathode crosser
                for part in inter.particles:
                    if part.shape != MICHL_SHP or part.muon_match_id == -1:
                        continue

                    muon = particles_by_id.get(part.muon_match_id)
                    if muon is None or not muon.is_cathode_crosser:
                        continue

                    offset = muon.cathode_offset
                    if not np.isfinite(offset) or offset == 0:
                        continue

                    points = self.get_points(part)
                    sources = self.get_sources(part)
                    if len(points) == 0:
                        continue

                    depositions = self.get_depositions(part).astype(np.float64)
                    corrected = depositions.copy()

                    # The Michel's points may span more than one chamber if
                    # it happens to straddle the cathode itself; handle each
                    # independently, as `cathode_crosser` does for the muon
                    chamber_ids = self.geo.get_chambers(sources)
                    for cid in np.unique(chamber_ids):
                        mask = chamber_ids == cid
                        mod = cid // self.geo.tpc.num_chambers_per_module
                        tpc = cid % self.geo.tpc.num_chambers_per_module
                        daxis = self.geo.tpc[mod][tpc].drift_axis
                        dsign = self.geo.tpc[mod][tpc].drift_sign
                        offset_t = dsign * offset

                        chamber_points = points[mask]
                        chamber_deps = corrected[mask]

                        if self.undo_original_calib:
                            # Reconstruct the pre-shift position, use it to
                            # find and divide out the stale correction factor
                            orig_points = chamber_points.copy()
                            orig_points[:, daxis] -= offset_t
                            undo_factor = self.calibrator.process(
                                orig_points,
                                np.ones(int(mask.sum())),
                                self.geo,
                                cid,
                                run_id,
                            )
                            chamber_deps = chamber_deps / undo_factor

                        # Reapply the correction at the corrected position
                        corrected[mask] = self.calibrator.process(
                            chamber_points, chamber_deps, self.geo, cid, run_id
                        )

                    part.corrected_depositions = corrected.astype(np.float32)
                    part.corrected_calo_ke = self.scaling * float(np.sum(corrected))
