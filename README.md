# LightFusion (LightGBM Inference on Datafusion)

LightFusion is an very opinionated [LightGBM](https://lightgbm.readthedocs.io/en/stable/)  datafusion integration, implemented to demonstrate datafusion `FunctionFactory` functionality merge request ([arrow-datafusion/pull#9333](https://github.com/apache/arrow-datafusion/pull/9333)).

> [!NOTE]
> It has not been envisaged as a actively maintained library.

Other project utilizing `FunctionFactory`:

- [Torchfusion, Opinionated Torch Inference on DataFusion](https://github.com/milenkovicm/torchfusion)
- [DataFusion JVM User Defined Functions (UDF)](https://github.com/milenkovicm/adhesive)

## How to use

A LightGBM model can be defined as and SQL UDF definition:

```sql
CREATE FUNCTION f0(DOUBLE[])
RETURNS DOUBLE[]
LANGUAGE LIGHTGBM
AS 'multiclass.lgbm'
```

and called from sql like:

```sql
SELECT f0([0.109, 1.261, -0.274, 2.605, 0.472, -0.429, -0.983, 1.000, -0.095, -1.219, -0.369, -0.312,
        -0.840, 1.281, -0.618, -0.532, -0.132, 0.443, 0.028, 2.201, 0.044, 1.671, 0.660, -0.114,
        0.574, 0.276, 0.680, -0.670]) as inferred
```

> [!WARNING]  
> Model used for tests could probably be way better.


## Configuration

FunctionFactor exposes set of configuration options which can be retrieved querying system catalog:

```text
+------------------------+-------+---------------------------------------------------------------------------+
| name                   | value | description                                                               |
+------------------------+-------+---------------------------------------------------------------------------+
| lightfusion.batch_size | 1     | Batch size to be used. Valid value positive non-zero integers. Default: 1 |
+------------------------+-------+---------------------------------------------------------------------------+
```

Available configuration options can be changed:

```sql
SET lightfusion.batch_size = 16
```
