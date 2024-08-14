use std::sync::Arc;

use datafusion::{
    arrow::datatypes::DataType,
    common::internal_err,
    execution::{
        config::SessionConfig,
        context::{FunctionFactory, RegisterFunction, SessionContext, SessionState},
        runtime_env::{RuntimeConfig, RuntimeEnv},
    },
    logical_expr::{CreateFunction, Expr, ScalarUDF},
    scalar::ScalarValue,
};
use log::debug;

mod argmax;
mod config;
mod udf;

pub use argmax::*;
pub use config::*;

#[derive(Default, Debug)]
pub struct LightfusionFunctionFactory {}

#[async_trait::async_trait]
impl FunctionFactory for LightfusionFunctionFactory {
    async fn create(
        &self,
        state: &SessionState,
        statement: CreateFunction,
    ) -> datafusion::error::Result<RegisterFunction> {
        let model_name = statement.name;

        let arg_data_type = statement
            .args
            .map(|a| {
                a.first()
                    .map(|r| r.data_type.clone())
                    .unwrap_or(DataType::Float64)
            })
            .unwrap_or(DataType::Float64);

        let data_type_input = find_item_type(&arg_data_type);

        let data_type_return = statement
            .return_type
            .map(|t| find_item_type(&t))
            .unwrap_or(data_type_input.clone());

        let model_file = match statement.params.function_body {
            Some(Expr::Literal(ScalarValue::Utf8(Some(s)))) => s,
            // we should handle this error better
            Some(e) => return internal_err!("Unsupported expression {e}")?,
            _ => format!("model/{}.lgbm", model_name),
        };
        let config = state
            .config()
            .options()
            .extensions
            .get::<LightfusionConfig>()
            .expect("function factory configuration to be set");

        // same device will be used untill function is dropped

        let batch_size = config.batch_size();
        let model_udf = udf::load_model(
            &model_name,
            &model_file,
            batch_size,
            data_type_input,
            data_type_return,
        )?;

        debug!("Registering function: [{:?}]", model_udf);

        Ok(RegisterFunction::Scalar(Arc::new(model_udf)))
    }
}

fn find_item_type(dtype: &DataType) -> DataType {
    match dtype {
        // We're interested in array type not the array.
        // There is discrepancy between array type defined by create function
        // `List(Field { name: \"field\", data_type: Float32, nullable:  ...``
        // and arry type defined by create array operation
        //`[List(Field { name: \"item\", data_type: Float64, nullable: true, ...`
        // so we just extract bits we need
        //
        // In general type handling is very optimistic
        // at the moment, but good enough for poc
        DataType::List(f) => f.data_type().clone(),
        r => r.clone(),
    }
}

pub fn configure_context() -> SessionContext {
    let runtime_environment = RuntimeEnv::new(RuntimeConfig::new()).unwrap();

    let session_config = SessionConfig::new()
        .with_information_schema(true)
        .with_option_extension(LightfusionConfig::default());
    let state = datafusion::execution::session_state::SessionStateBuilder::new()
        .with_config(session_config)
        .with_runtime_env(runtime_environment.into())
        .with_default_features()
        .with_function_factory(Some(Arc::new(LightfusionFunctionFactory::default())))
        .build();

    let ctx = SessionContext::new_with_state(state);

    ctx.register_udf(ScalarUDF::from(crate::ArgMax::new()));

    ctx
}
#[cfg(test)]
mod test {
    use datafusion::{assert_batches_eq, error::Result};

    #[tokio::test]
    async fn e2e() -> Result<()> {
        let ctx = crate::configure_context();

        let sql = r#"
        CREATE FUNCTION f0(DOUBLE[])
        RETURNS DOUBLE[]
        LANGUAGE LIGHTGBM
        AS 'multiclass.lgbm'
        "#;

        ctx.sql(sql).await?.collect().await?;

        let sql = r#"
        SELECT f0([0.109, 1.261, -0.274, 2.605, 0.472, -0.429, -0.983, 1.000, -0.095, -1.219, -0.369, -0.312,
            -0.840, 1.281, -0.618, -0.532, -0.132, 0.443, 0.028, 2.201, 0.044, 1.671, 0.660, -0.114,
            0.574, 0.276, 0.680, -0.670]) as inferred
        "#;

        let expected = vec![
        "+----------------------------------------------------------------------------------------------------------+", 
        "| inferred                                                                                                 |", 
        "+----------------------------------------------------------------------------------------------------------+", 
        "| [0.21591763144158005, 0.3469195372175628, 0.33502631180449255, 0.03381740990463416, 0.06831910963173049] |", 
        "+----------------------------------------------------------------------------------------------------------+"]
        ;

        let result = ctx.sql(sql).await?.collect().await?;
        assert_batches_eq!(expected, &result);

        let sql = r#"
        SELECT argmax(f0([0.109, 1.261, -0.274, 2.605, 0.472, -0.429, -0.983, 1.000, -0.095, -1.219, -0.369, -0.312,
            -0.840, 1.281, -0.618, -0.532, -0.132, 0.443, 0.028, 2.201, 0.044, 1.671, 0.660, -0.114,
            0.574, 0.276, 0.680, -0.670])) as inferred
        "#;

        let expected = vec![
            "+----------+",
            "| inferred |",
            "+----------+",
            "| 1        |",
            "+----------+",
        ];

        let result = ctx.sql(sql).await?.collect().await?;
        assert_batches_eq!(expected, &result);

        Ok(())
    }
}

#[cfg(test)]
#[ctor::ctor]
fn init() {
    // Enable RUST_LOG logging configuration for tests
    let _ = env_logger::builder().is_test(true).try_init();
}
