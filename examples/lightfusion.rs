#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    let ctx = lightfusion::configure_context();
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
    datafusion::assert_batches_eq!(expected, &result);

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
    datafusion::assert_batches_eq!(expected, &result);

    ctx.sql("SELECT * FROM information_schema.df_settings WHERE NAME LIKE 'lightfusion%'")
        .await?
        .show()
        .await?;

    Ok(())
}
