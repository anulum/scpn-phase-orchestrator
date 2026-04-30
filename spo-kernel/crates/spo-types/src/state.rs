// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — State types

use std::collections::HashSet;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::action::Regime;
use crate::error::{SpoError, SpoResult};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Channel {
    P,
    I,
    S,
    Custom(String),
}

impl Channel {
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        match id.into().as_str() {
            "P" => Self::P,
            "I" => Self::I,
            "S" => Self::S,
            other => Self::Custom(other.to_owned()),
        }
    }

    /// Builds a channel from a runtime identifier.
    ///
    /// # Errors
    ///
    /// Returns [`SpoError::InvalidConfig`] when the identifier is empty or
    /// contains only whitespace.
    pub fn try_new(id: impl Into<String>) -> SpoResult<Self> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(SpoError::InvalidConfig(
                "channel identifier must not be empty".into(),
            ));
        }
        Ok(Self::new(id))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::P => "P",
            Self::I => "I",
            Self::S => "S",
            Self::Custom(id) => id,
        }
    }

    #[must_use]
    pub fn is_builtin(&self) -> bool {
        matches!(self, Self::P | Self::I | Self::S)
    }
}

impl Serialize for Channel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Channel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let id = String::deserialize(deserializer)?;
        Self::try_new(id).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelMetric {
    pub channel: Channel,
    pub r: f64,
    pub psi: f64,
    pub weight: f64,
}

impl ChannelMetric {
    /// Builds and validates one channel-level coherence metric.
    ///
    /// # Errors
    ///
    /// Returns [`SpoError::InvalidConfig`] when `r` is non-finite or outside
    /// `[0, 1]`, `psi` is non-finite, or `weight` is non-finite/negative.
    pub fn new(channel: Channel, r: f64, psi: f64, weight: f64) -> SpoResult<Self> {
        let metric = Self {
            channel,
            r,
            psi,
            weight,
        };
        metric.validate()?;
        Ok(metric)
    }

    /// Validates finite ranges for one channel-level coherence metric.
    ///
    /// # Errors
    ///
    /// Returns [`SpoError::InvalidConfig`] when `r` is non-finite or outside
    /// `[0, 1]`, `psi` is non-finite, or `weight` is non-finite/negative.
    pub fn validate(&self) -> SpoResult<()> {
        if !self.r.is_finite() || !(0.0..=1.0).contains(&self.r) {
            return Err(SpoError::InvalidConfig(format!(
                "channel {}: r must be finite and in [0, 1]",
                self.channel.as_str()
            )));
        }
        if !self.psi.is_finite() {
            return Err(SpoError::InvalidConfig(format!(
                "channel {}: psi must be finite",
                self.channel.as_str()
            )));
        }
        if !self.weight.is_finite() || self.weight < 0.0 {
            return Err(SpoError::InvalidConfig(format!(
                "channel {}: weight must be finite and non-negative",
                self.channel.as_str()
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhaseState {
    pub theta: f64,
    pub omega: f64,
    pub amplitude: f64,
    pub quality: f64,
    pub channel: Channel,
    pub node_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerState {
    pub r: f64,
    pub psi: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UPDEState {
    pub layers: Vec<LayerState>,
    #[serde(default)]
    pub channel_metrics: Vec<ChannelMetric>,
    pub cross_layer_alignment: Vec<f64>,
    pub stability_proxy: f64,
    pub regime: Regime,
}

impl UPDEState {
    #[must_use]
    pub fn mean_r(&self) -> f64 {
        if self.layers.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.layers.iter().map(|l| l.r).sum();
        sum / self.layers.len() as f64
    }

    #[must_use]
    pub fn channel_metric(&self, channel: &Channel) -> Option<&ChannelMetric> {
        self.channel_metrics
            .iter()
            .find(|metric| &metric.channel == channel)
    }

    #[must_use]
    pub fn mean_channel_r(&self, channels: &[Channel]) -> f64 {
        let vals: Vec<f64> = channels
            .iter()
            .filter_map(|channel| self.channel_metric(channel).map(|metric| metric.r))
            .collect();
        if vals.is_empty() {
            return 0.0;
        }
        vals.iter().sum::<f64>() / vals.len() as f64
    }

    /// Validates all channel-level metrics in the state.
    ///
    /// # Errors
    ///
    /// Returns [`SpoError::InvalidConfig`] when any metric has invalid numeric
    /// ranges or when multiple metrics use the same channel identifier.
    pub fn validate_channel_metrics(&self) -> SpoResult<()> {
        let mut seen = HashSet::new();
        for metric in &self.channel_metrics {
            metric.validate()?;
            if !seen.insert(metric.channel.clone()) {
                return Err(SpoError::InvalidConfig(format!(
                    "duplicate channel metric for {}",
                    metric.channel.as_str()
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_serde() {
        let json = serde_json::to_string(&Channel::S).expect("serialise channel");
        let c: Channel = serde_json::from_str(&json).expect("deserialise channel");
        assert_eq!(c, Channel::S);
    }

    #[test]
    fn custom_channel_serde() {
        let channel = Channel::new("Risk");
        assert_eq!(channel.as_str(), "Risk");
        assert!(!channel.is_builtin());
        let json = serde_json::to_string(&channel).expect("serialise custom channel");
        assert_eq!(json, "\"Risk\"");
        let c: Channel = serde_json::from_str(&json).expect("deserialise custom channel");
        assert_eq!(c, Channel::Custom("Risk".into()));
    }

    #[test]
    fn empty_custom_channel_rejected() {
        let err = serde_json::from_str::<Channel>("\" \"")
            .expect_err("empty channel identifier must be rejected");
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn try_new_rejects_empty_channel() {
        let err = Channel::try_new("\t").expect_err("empty channel identifier must be rejected");
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn phase_state_fields() {
        let ps = PhaseState {
            theta: 1.0,
            omega: 2.0,
            amplitude: 0.5,
            quality: 0.9,
            channel: Channel::P,
            node_id: "n0".into(),
        };
        assert_eq!(ps.theta, 1.0);
        assert_eq!(ps.channel, Channel::P);
    }

    #[test]
    fn phase_state_accepts_custom_channel() {
        let ps = PhaseState {
            theta: 1.0,
            omega: 2.0,
            amplitude: 0.5,
            quality: 0.9,
            channel: Channel::new("Mission"),
            node_id: "n0".into(),
        };
        let json = serde_json::to_string(&ps).expect("serialise phase state");
        assert!(json.contains("\"channel\":\"Mission\""));
        let roundtrip: PhaseState = serde_json::from_str(&json).expect("deserialise phase state");
        assert_eq!(roundtrip.channel.as_str(), "Mission");
    }

    #[test]
    fn layer_state_serde() {
        let ls = LayerState { r: 0.8, psi: 1.5 };
        let json = serde_json::to_string(&ls).expect("serialise layer state");
        let ls2: LayerState = serde_json::from_str(&json).expect("deserialise layer state");
        assert!((ls2.r - 0.8).abs() < 1e-12);
    }

    #[test]
    fn channel_metric_validates_ranges() {
        let metric =
            ChannelMetric::new(Channel::new("Risk"), 0.7, 1.5, 0.25).expect("valid channel metric");
        assert_eq!(metric.channel.as_str(), "Risk");
        assert!((metric.r - 0.7).abs() < 1e-12);

        let err = ChannelMetric::new(Channel::I, 1.1, 0.0, 1.0)
            .expect_err("r outside [0, 1] must be rejected");
        assert!(err.to_string().contains("r must be finite"));

        let err = ChannelMetric::new(Channel::S, 0.5, f64::NAN, 1.0)
            .expect_err("non-finite psi must be rejected");
        assert!(err.to_string().contains("psi must be finite"));

        let err = ChannelMetric::new(Channel::P, 0.5, 0.0, -0.1)
            .expect_err("negative weight must be rejected");
        assert!(err.to_string().contains("weight must be finite"));
    }

    #[test]
    fn upde_state_mean_r() {
        let state = UPDEState {
            layers: vec![
                LayerState { r: 0.4, psi: 0.0 },
                LayerState { r: 0.8, psi: 0.0 },
            ],
            channel_metrics: vec![],
            cross_layer_alignment: vec![],
            stability_proxy: 0.5,
            regime: Regime::Nominal,
        };
        assert!((state.mean_r() - 0.6).abs() < 1e-12);
    }

    #[test]
    fn upde_state_mean_r_empty() {
        let state = UPDEState {
            layers: vec![],
            channel_metrics: vec![],
            cross_layer_alignment: vec![],
            stability_proxy: 0.0,
            regime: Regime::Nominal,
        };
        assert_eq!(state.mean_r(), 0.0);
    }

    #[test]
    fn upde_state_channel_metrics_are_n_channel() {
        let state = UPDEState {
            layers: vec![],
            channel_metrics: vec![
                ChannelMetric::new(Channel::P, 0.8, 0.0, 1.0).expect("valid P metric"),
                ChannelMetric::new(Channel::new("Risk"), 0.2, 1.0, 0.5)
                    .expect("valid custom metric"),
                ChannelMetric::new(Channel::new("Mission"), 0.6, 2.0, 0.25)
                    .expect("valid custom metric"),
            ],
            cross_layer_alignment: vec![],
            stability_proxy: 0.0,
            regime: Regime::Nominal,
        };

        assert!(state.validate_channel_metrics().is_ok());
        assert_eq!(
            state
                .channel_metric(&Channel::new("Risk"))
                .expect("Risk metric")
                .channel
                .as_str(),
            "Risk"
        );
        assert!(
            (state.mean_channel_r(&[Channel::new("Risk"), Channel::new("Mission")]) - 0.4).abs()
                < 1e-12
        );
    }

    #[test]
    fn upde_state_rejects_duplicate_channel_metrics() {
        let state = UPDEState {
            layers: vec![],
            channel_metrics: vec![
                ChannelMetric::new(Channel::new("Risk"), 0.2, 1.0, 0.5)
                    .expect("valid custom metric"),
                ChannelMetric::new(Channel::new("Risk"), 0.3, 2.0, 0.25)
                    .expect("valid custom metric"),
            ],
            cross_layer_alignment: vec![],
            stability_proxy: 0.0,
            regime: Regime::Nominal,
        };

        let err = state
            .validate_channel_metrics()
            .expect_err("duplicate channel metrics must be rejected");
        assert!(err.to_string().contains("duplicate channel metric"));
    }

    #[test]
    fn upde_state_deserialises_legacy_json_without_channel_metrics() {
        let json = r#"{
            "layers": [{"r": 0.5, "psi": 0.0}],
            "cross_layer_alignment": [],
            "stability_proxy": 0.5,
            "regime": "Nominal"
        }"#;

        let state: UPDEState = serde_json::from_str(json).expect("deserialise legacy UPDE state");

        assert!(state.channel_metrics.is_empty());
        assert_eq!(state.mean_r(), 0.5);
    }
}
