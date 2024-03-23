use datafusion::{
    config::{ConfigEntry, ConfigExtension, ExtensionOptions},
    error::DataFusionError,
};

#[derive(Debug, Clone)]
pub struct LightfusionConfig {
    batch_size: usize,
}

impl Default for LightfusionConfig {
    fn default() -> Self {
        Self { batch_size: 1 }
    }
}

impl ExtensionOptions for LightfusionConfig {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn cloned(&self) -> Box<dyn ExtensionOptions> {
        Box::new(self.clone())
    }

    fn set(&mut self, key: &str, value: &str) -> datafusion::error::Result<()> {
        match key.to_lowercase().as_str() {
            "batch_size" => {
                self.batch_size = value.parse().map_err(|_| {
                    DataFusionError::Configuration("batch size not correct".to_string())
                })?
            }
            key => Err(DataFusionError::Configuration(format!(
                "No configuration key: {key}"
            )))?,
        }

        Ok(())
    }

    fn entries(&self) -> Vec<datafusion::config::ConfigEntry> {
        vec![ConfigEntry {
            key: format!("{}.batch_size", Self::PREFIX),
            value: Some(format!("{}", self.batch_size)),
            description:
                "Batch size to be used. Valid value positive non-zero integers. Default: 1",
        }]
    }
}

impl LightfusionConfig {
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

impl ConfigExtension for LightfusionConfig {
    const PREFIX: &'static str = "lightfusion";
}
