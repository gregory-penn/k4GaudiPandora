/*
 * Copyright (c) 2020-2024 Key4hep-Project.
 *
 * This file is part of Key4hep.
 * See https://key4hep.github.io/key4hep-doc/ for further info.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DDMCParticleCreator.h"

#include <Api/PandoraApi.h>
#include <Objects/Helix.h>

#include <edm4hep/CaloHitContribution.h>
#include <edm4hep/CaloHitMCParticleLinkCollection.h>
#include <edm4hep/MCParticle.h>
#include <edm4hep/SimCalorimeterHit.h>
#include <edm4hep/SimTrackerHit.h>
#include <edm4hep/Track.h>
#include <edm4hep/TrackMCParticleLinkCollection.h>

#include <podio/ObjectID.h>

#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

// forward declarations. See in DDPandoraPFANewProcessor.cc
double getFieldFromCompact();

DDMCParticleCreator::DDMCParticleCreator(const Settings& settings, pandora::Pandora& pandora,
                                         const Gaudi::Algorithm* algorithm)
    : m_settings(settings), m_pandora(pandora), m_bField(getFieldFromCompact()), m_algorithm(*algorithm) {}

pandora::StatusCode
DDMCParticleCreator::CreateMCParticles(const std::vector<edm4hep::MCParticle>& mcParticleCollections) const {

  // Pandora identifies each MCParticle by the address of its element in mcParticleCollections. A daughter returned by
  // getDaughters() is a different object at a different address, so look up its element by ObjectID instead.
  std::unordered_map<podio::ObjectID, const edm4hep::MCParticle*> addressOf;
  for (const auto& mcParticle : mcParticleCollections) {
    addressOf.emplace(mcParticle.getObjectID(), &mcParticle);
    m_algorithm.debug() << "addressOf: ObjectID " << mcParticle.getObjectID() << " (PDG " << mcParticle.getPDG()
                        << ") -> " << &mcParticle << " (address given to Pandora)" << endmsg;
  }

  unsigned int nRelationships(0), nDaughtersNotFound(0);

  for (const auto& mcParticle : mcParticleCollections) {
    PandoraApi::MCParticle::Parameters mcParticleParameters;
    mcParticleParameters.m_energy = mcParticle.getEnergy();
    mcParticleParameters.m_particleId = mcParticle.getPDG();
    mcParticleParameters.m_mcParticleType = pandora::MC_3D;
    mcParticleParameters.m_pParentAddress = &mcParticle;
    mcParticleParameters.m_momentum =
        pandora::CartesianVector(mcParticle.getMomentum().x, mcParticle.getMomentum().y, mcParticle.getMomentum().z);
    mcParticleParameters.m_vertex =
        pandora::CartesianVector(mcParticle.getVertex().x, mcParticle.getVertex().y, mcParticle.getVertex().z);
    mcParticleParameters.m_endpoint =
        pandora::CartesianVector(mcParticle.getEndpoint().x, mcParticle.getEndpoint().y, mcParticle.getEndpoint().z);

    PANDORA_THROW_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=,
                            PandoraApi::MCParticle::Create(m_pandora, mcParticleParameters))

    // Create parent-daughter relationships
    for (const auto& daughter : mcParticle.getDaughters()) {
      const auto iter = addressOf.find(daughter.getObjectID());

      if (addressOf.end() == iter) {
        m_algorithm.debug() << "parent " << &mcParticle << " -> daughter ObjectID " << daughter.getObjectID()
                            << ": old address " << &daughter << ", not among the input MCParticles, skipped" << endmsg;
        ++nDaughtersNotFound;
        continue;
      }

      m_algorithm.debug() << "parent " << &mcParticle << " -> daughter ObjectID " << daughter.getObjectID()
                          << " (PDG " << daughter.getPDG() << "): old address " << &daughter << ", new address "
                          << iter->second << endmsg;

      PANDORA_THROW_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=,
                              PandoraApi::SetMCParentDaughterRelationship(m_pandora, &mcParticle, iter->second))
      ++nRelationships;
    }
  }

  m_algorithm.debug() << "MCParticle parent-daughter relationships: " << nRelationships << " set, "
                      << nDaughtersNotFound << " daughters not among the input MCParticles" << endmsg;

  return pandora::STATUS_CODE_SUCCESS;
}

pandora::StatusCode
DDMCParticleCreator::CreateTrackToMCParticleRelationships(const TrackMCLinkCollectionVector& trackRelCollections,
                                                          const TrackVector& trackVector) const {
  throw std::runtime_error("DDMCParticleCreator::CreateTrackToMCParticleRelationships not implemented yet");
  for (const auto& trackRelCollection : trackRelCollections) {
    for (const auto& pTrack : trackVector) {
      try {
        // Get reconstructed momentum at dca
        const auto& trackState = pTrack.getTrackStates(0);
        const pandora::Helix helixFit(trackState.phi, trackState.D0, trackState.Z0, trackState.omega,
                                      trackState.tanLambda, m_bField);
        const float recoMomentum = helixFit.GetMomentum().GetMagnitude();

        // APS: I am not sure why we chose the best MCParticle per trackRelationCollection, but this was like this from
        // the start, and we only have one track collection, so it doesn't matter...

        // Use momentum magnitude to identify best MC particle
        const edm4hep::MCParticle* pBestMCParticle = nullptr;
        float bestDeltaMomentum = std::numeric_limits<float>::max();

        for (const auto&& trackMCRel : *trackRelCollection) {
          // if the track relation does not match the track, continue
          if (trackMCRel.get<edm4hep::Track>().id() != pTrack.id())
            continue;

          auto const& mcParticle = trackMCRel.get<edm4hep::MCParticle>();

          const float trueMomentum = pandora::CartesianVector(mcParticle.getMomentum()[0], mcParticle.getMomentum()[1],
                                                              mcParticle.getMomentum()[2])
                                         .GetMagnitude();
          const float deltaMomentum = std::fabs(recoMomentum - trueMomentum);

          if (deltaMomentum < bestDeltaMomentum) {
            pBestMCParticle = &mcParticle;
            bestDeltaMomentum = deltaMomentum;
          }
        }

        if (pBestMCParticle == nullptr) {
          m_algorithm.warning() << "No suitable MC particle found for track association." << endmsg;
          continue;
        }

        PANDORA_THROW_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=,
                                PandoraApi::SetTrackToMCParticleRelationship(m_pandora, &pTrack, pBestMCParticle))

      } catch (const pandora::StatusCodeException& statusCodeException) {
        m_algorithm.error() << "Failed to extract track to MC particle relationship: " << statusCodeException.ToString()
                            << endmsg;
      } catch (const std::exception& exception) {
        m_algorithm.warning() << "Exception encountered: " << exception.what() << endmsg;
      }
    }
  }

  return pandora::STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

pandora::StatusCode DDMCParticleCreator::CreateCaloHitToMCParticleRelationships(
    const CaloHitSimCaloHitLinkCollectionVector& caloRelCollections,
    const std::vector<const HitVector*>& calorimeterHitVectors,
    const std::vector<edm4hep::MCParticle>& mcParticleVector) const {

  // Table A, hits. Pandora identifies each calorimeter hit by the address of its element in one of the hit vectors
  // that DDPandoraPFANewAlgorithm handed to the calo hit creator (getCommonCaloHitProperties: m_pParentAddress = &hit).
  // A hit returned by a link is a different object at a different address, so look up its element by the hit itself.
  // The key is the edm4hep::CalorimeterHit handle: its std::hash and operator== use the address of the hit's stored
  // data (m_obj), which the handle in the vector and the handle returned by the link share. The ObjectID cannot be the
  // key: hit collections made during the job have no collection ID yet (ffffffff), so ECAL hit 0 and HCAL hit 0 would
  // both have ObjectID ffffffff|0, and emplace would keep only the first of them.
  std::unordered_map<edm4hep::CalorimeterHit, const edm4hep::CalorimeterHit*> hitAddressOf;
  for (const HitVector* const pHitVector : calorimeterHitVectors) {
    for (const auto& caloHit : *pHitVector)
      hitAddressOf.emplace(caloHit, &caloHit);
  }

  // Table B, MCParticles. Same as in CreateMCParticles: ObjectID -> address of the element given to Pandora.
  std::unordered_map<podio::ObjectID, const edm4hep::MCParticle*> particleAddressOf;
  for (const auto& mcParticle : mcParticleVector)
    particleAddressOf.emplace(mcParticle.getObjectID(), &mcParticle);

  m_algorithm.debug() << "CaloHit truth: " << hitAddressOf.size() << " hits and " << particleAddressOf.size()
                      << " MCParticles in the lookup tables" << endmsg;

  unsigned int nLinks(0), nHitsNotFound(0), nParticlesNotFound(0), nRelationships(0);

  // Each link connects one CalorimeterHit (its "from" side) to one SimCalorimeterHit (its "to" side)
  for (const auto* const pLinkCollection : caloRelCollections) {
    for (const auto& caloHitLink : *pLinkCollection) {
      ++nLinks;

      try {
        // Step 1: find the address Pandora was given for this hit
        const edm4hep::CalorimeterHit caloHit = caloHitLink.getFrom();
        const auto hitIter = hitAddressOf.find(caloHit);

        // "hit key" is the value the table looks the hit up by: std::hash<edm4hep::CalorimeterHit>, which hashes the
        // address of the hit's stored data (m_obj). With libstdc++ the hash of an address is the address itself.
        const auto hitKey = std::hash<edm4hep::CalorimeterHit>{}(caloHit);

        if (hitAddressOf.end() == hitIter) {
          m_algorithm.debug() << "link " << nLinks << ": hit key 0x" << std::hex << hitKey << std::dec
                              << " was not given to Pandora, skipped" << endmsg;
          ++nHitsNotFound;
          continue;
        }

        // Step 2: the simulated hit holds the true energy contributions, one per (MCParticle, Geant4 step)
        const edm4hep::SimCalorimeterHit simHit = caloHitLink.getTo();
        m_algorithm.debug() << "link " << nLinks << ": hit key 0x" << std::hex << hitKey << std::dec << " -> address "
                            << hitIter->second << ", sim hit ObjectID " << simHit.getObjectID() << " with "
                            << simHit.getContributions().size() << " contributions" << endmsg;

        // Energies of this hit: the reconstructed hit (after digitisation and calibration) and the simulated hit (true
        // deposited energy, which should equal the sum of its contributions printed below)
        m_algorithm.debug() << "    hit energy: reconstructed " << caloHit.getEnergy() << " GeV, simulated "
                            << simHit.getEnergy() << " GeV" << endmsg;

        // Steps 3 and 4: sum the true energy per MCParticle, keyed by the address Pandora was given for that particle.
        // "old address" is what the previous code used (&ipa): the address of the local copy returned by getParticle().
        std::map<const edm4hep::MCParticle*, float> energyByParticle;
        float contributionEnergySum(0.f);
        for (const auto& contribution : simHit.getContributions()) {
          contributionEnergySum += contribution.getEnergy();
          const edm4hep::MCParticle particle = contribution.getParticle();
          const auto particleIter = particleAddressOf.find(particle.getObjectID());

          if (particleAddressOf.end() == particleIter) {
            m_algorithm.debug() << "    contribution from particle ObjectID " << particle.getObjectID() << ", "
                                << contribution.getEnergy() << " GeV, not among the input MCParticles, skipped"
                                << endmsg;
            ++nParticlesNotFound;
            continue;
          }

          m_algorithm.debug() << "    contribution from particle ObjectID " << particle.getObjectID() << " (PDG "
                              << particle.getPDG() << "), " << contribution.getEnergy() << " GeV: old address "
                              << &particle << ", new address " << particleIter->second << endmsg;
          energyByParticle[particleIter->second] += contribution.getEnergy();
        }

        // Sum over all contributions, including skipped ones; compare with the simulated hit energy printed above
        m_algorithm.debug() << "    sum of contributions: " << contributionEnergySum << " GeV (simulated hit energy "
                            << simHit.getEnergy() << " GeV)" << endmsg;

        // Step 5: one relationship per (hit, MCParticle), weighted by the summed true energy. If several links point at
        // the same hit, Pandora adds the weights for the same particle (MCManager::SetUidToMCParticleRelationship).
        for (const auto& [pParticle, energy] : energyByParticle) {
          m_algorithm.debug() << "    -> relationship: hit " << hitIter->second << ", particle " << pParticle << ", "
                              << energy << " GeV" << endmsg;
          PANDORA_THROW_RESULT_IF(
              pandora::STATUS_CODE_SUCCESS, !=,
              PandoraApi::SetCaloHitToMCParticleRelationship(m_pandora, hitIter->second, pParticle, energy))
          ++nRelationships;
        }
      } catch (const pandora::StatusCodeException& statusCodeException) {
        m_algorithm.error() << "Failed to extract calo hit to mc particle relationship: "
                            << statusCodeException.ToString() << endmsg;
      } catch (const std::exception& exception) {
        m_algorithm.warning() << "Failed to extract calo hit to mc particle relationship: " << exception.what()
                              << endmsg;
      }
    }
  }

  m_algorithm.debug() << "CaloHit truth summary: " << nLinks << " links read, " << nHitsNotFound
                      << " with a hit not given to Pandora, " << nParticlesNotFound
                      << " contributions from MCParticles not among the input MCParticles, " << nRelationships
                      << " hit to MCParticle relationships set" << endmsg;

  return pandora::STATUS_CODE_SUCCESS;
}

DDMCParticleCreator::Settings::Settings()
    : m_mcParticleCollections(StringVector()), m_caloHitRelationCollections(StringVector()),
      m_trackRelationCollections(StringVector()) {}
