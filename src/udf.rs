use datafusion::{
    arrow::{
        array::{Array, ArrayBuilder, ArrayRef, ListArray, PrimitiveArray, PrimitiveBuilder},
        buffer::{OffsetBuffer, ScalarBuffer},
        datatypes::{ArrowPrimitiveType, DataType, Field, Float32Type, Float64Type},
    },
    common::{downcast_value, exec_err},
    error::{DataFusionError, Result},
    logical_expr::{ColumnarValue, ScalarUDF, ScalarUDFImpl, Signature, Volatility},
};
use debug_ignore::DebugIgnore;
use lightgbm3::{Booster, DType};
use std::{any::Any, fmt::Debug, marker::PhantomData, sync::Arc};

pub fn load_model(
    model_name: &str,
    model_file: &str,
    batch_size: usize,
    input_type: DataType,
    return_type: DataType,
) -> Result<ScalarUDF> {
    match (input_type, return_type) {
        (DataType::Float32, DataType::Float64) => {
            let model_udf = LightfusionUdf::<Float32Type, Float64Type>::new_from_file(
                model_name, model_file, batch_size,
            )?;
            Ok(ScalarUDF::from(model_udf))
        }
        (DataType::Float64, DataType::Float64) => {
            let model_udf = LightfusionUdf::<Float64Type, Float64Type>::new_from_file(
                model_name, model_file, batch_size,
            )?;
            Ok(ScalarUDF::from(model_udf))
        }
        (DataType::Float32 | DataType::Float64, t) => exec_err!(
            "Return type not supported : {}, expected: {}",
            t,
            DataType::Float64
        )?,
        (t, _) => exec_err!(
            "Input type not supported : {}, expected: {} or {}",
            t,
            DataType::Float32,
            DataType::Float64
        )?,
    }
}

#[derive(Debug)]
struct LightfusionUdf<I, R>
where
    I: ArrowPrimitiveType + Debug,
    R: ArrowPrimitiveType + Debug,
{
    name: String,
    batch_size: usize,
    booster: DebugIgnore<Booster>,
    signature: Signature,
    return_type_filed: Arc<Field>,
    phantom_i: PhantomData<I>,
    phantom_r: PhantomData<R>,
}

impl<I, R> LightfusionUdf<I, R>
where
    I: ArrowPrimitiveType + Debug,
    R: ArrowPrimitiveType + Debug,
{
    fn new_from_file(name: &str, model_file: &str, batch_size: usize) -> Result<Self> {
        let return_type_filed = Arc::new(Field::new("item", R::DATA_TYPE.clone(), false));

        let signature = Signature::exact(
            vec![DataType::List(Arc::new(Field::new(
                "item",
                I::DATA_TYPE.clone(),
                false,
            )))],
            Volatility::Immutable,
        );
        let booster = Self::load_model_from(model_file)?;

        Ok(Self {
            name: name.to_string(),
            batch_size,
            booster: booster.into(),
            signature,
            return_type_filed,
            phantom_i: PhantomData,
            phantom_r: PhantomData,
        })
    }
    fn load_model_from(model_file: &str) -> Result<Booster> {
        let booster = Booster::from_file(model_file)
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;

        Ok(booster)
    }
}

// should be safe as we do not mutate it
unsafe impl<I, R> Sync for LightfusionUdf<I, R>
where
    I: ArrowPrimitiveType + Debug,
    R: ArrowPrimitiveType + Debug,
{
}
unsafe impl<I, R> Send for LightfusionUdf<I, R>
where
    I: ArrowPrimitiveType + Debug,
    R: ArrowPrimitiveType + Debug,
{
}

impl<I, R> ScalarUDFImpl for LightfusionUdf<I, R>
where
    I: ArrowPrimitiveType + Debug + Send + Sync,
    R: ArrowPrimitiveType<Native = f64> + Debug + Send + Sync,
    <I as ArrowPrimitiveType>::Native: DType,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::List(self.return_type_filed.clone()))
    }

    fn invoke(&self, args: &[ColumnarValue]) -> Result<ColumnarValue> {
        let args = ColumnarValue::values_to_arrays(args)?;
        let features = datafusion::common::cast::as_list_array(&args[0])?;

        let offsets = features.offsets();

        let (result_offsets, values) = {
            let values = downcast_value!(features.values(), PrimitiveArray, I);
            Self::_call_model(
                PrimitiveBuilder::<R>::new(),
                values,
                offsets,
                &self.booster,
                self.batch_size,
            )?
        };
        let array = ListArray::new(self.return_type_filed.clone(), result_offsets, values, None);

        Ok(ColumnarValue::from(Arc::new(array) as ArrayRef))
    }
}

impl<I, R> LightfusionUdf<I, R>
where
    I: ArrowPrimitiveType + Debug,
    R: ArrowPrimitiveType<Native = f64> + Debug,
    <I as ArrowPrimitiveType>::Native: DType,
{
    #[inline]
    fn _call_model(
        mut result: PrimitiveBuilder<R>,
        values: &PrimitiveArray<I>,
        offsets: &OffsetBuffer<i32>,
        gbm: &Booster,

        batch_size: usize,
    ) -> Result<(
        OffsetBuffer<i32>,
        Arc<(dyn datafusion::arrow::array::Array + 'static)>,
    )>
where {
        let mut result_offsets: Vec<i32> = vec![];
        result_offsets.push(0);
        let mut start = 0;

        while let Some((tensor, next_start, no_items)) =
            // this would make more sense if we return iterator
            Self::create_batched_tensor(start, batch_size, values, offsets)
        {
            start = next_start;
            let n_features = tensor.len() / no_items;
            let logits = gbm
                .predict(tensor, n_features as i32, batch_size > 1)
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;

            Self::flatten_batched_tensor(&logits[..], no_items, &mut result_offsets, &mut result)?;
        }

        Ok((
            OffsetBuffer::new(ScalarBuffer::from(result_offsets)),
            Arc::new(result.finish()) as ArrayRef,
        ))
    }

    fn create_batched_tensor<'a, T>(
        start_offset: usize,
        batch_size: usize,
        values: &'a PrimitiveArray<T>,
        offsets: &OffsetBuffer<i32>,
    ) -> Option<(&'a [T::Native], usize, usize)>
    where
        T: ArrowPrimitiveType,
    {
        let end_offset = std::cmp::min(start_offset + batch_size, offsets.len() - 1);
        if end_offset <= start_offset {
            None
        } else {
            let index_start = offsets[start_offset] as usize;
            let index_end = offsets[end_offset] as usize;
            let total_items = end_offset - start_offset;

            let current: &[T::Native] = &values.values()[index_start..index_end];

            Some((current, end_offset, total_items))
        }
    }

    fn flatten_batched_tensor(
        logits: &[R::Native],
        items: usize,
        result_offsets: &mut Vec<i32>,
        result: &mut PrimitiveBuilder<R>,
    ) -> Result<()>
where {
        let start = result.len();

        result.append_slice(logits);
        let end = result.len();
        let elements = (end - start) / items;

        // populate resulting offsets from result
        (1..=items).for_each(|i| result_offsets.push((start + i * elements) as i32));

        Ok(())
    }
}
