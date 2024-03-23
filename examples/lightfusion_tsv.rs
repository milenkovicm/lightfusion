#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    let ctx = lightfusion::configure_context();

    let sql = r#"
    SET lightfusion.batch_size = 1
    "#;

    ctx.sql(sql).await?.collect().await?;

    let sql = r#"
    CREATE FUNCTION f0(DOUBLE[])
    RETURNS DOUBLE[]
    LANGUAGE LIGHTGBM
    AS 'multiclass.lgbm'
    "#;

    ctx.sql(sql).await?.collect().await?;

    let sql = r#"
    CREATE EXTERNAL TABLE m STORED AS CSV DELIMITER '	' LOCATION 'multiclass.test'
    "#;

    ctx.sql(sql).await?.collect().await?;

    let sql = r#"
    SELECT column_1 as label, 
    argmax(
        f0(
            [
            column_2,
            column_3,
            column_4,
            column_5,
            column_6,
            column_7,
            column_8,
            column_9,
            column_10,
            column_11,
            column_12,
            column_13,
            column_14,
            column_15,
            column_16,
            column_17,
            column_18,
            column_19,
            column_20,
            column_21,
            column_22,
            column_23,
            column_24,
            column_25,
            column_26,
            column_27,
            column_28,
            column_29
            ]
        )
    ) as inferred
     FROM m
    "#;

    ctx.sql(sql).await?.show().await?;

    Ok(())
}
