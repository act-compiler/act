#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Dtype {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    FP16,
    FP32,
    FP64,
    BF16,
    Pred,
}

impl Dtype {
    pub fn size_in_bytes(&self) -> i32 {
        match self {
            Dtype::U8 => 1,
            Dtype::I8 => 1,
            Dtype::U16 => 2,
            Dtype::I16 => 2,
            Dtype::U32 => 4,
            Dtype::I32 => 4,
            Dtype::U64 => 8,
            Dtype::I64 => 8,
            Dtype::FP16 => 2,
            Dtype::FP32 => 4,
            Dtype::FP64 => 8,
            Dtype::BF16 => 2,
            Dtype::Pred => 1,
        }
    }
}

impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dtype::U8 => write!(f, "u8"),
            Dtype::I8 => write!(f, "s8"),
            Dtype::U16 => write!(f, "u16"),
            Dtype::I16 => write!(f, "s16"),
            Dtype::U32 => write!(f, "u32"),
            Dtype::I32 => write!(f, "s32"),
            Dtype::U64 => write!(f, "u64"),
            Dtype::I64 => write!(f, "s64"),
            Dtype::FP16 => write!(f, "f16"),
            Dtype::FP32 => write!(f, "f32"),
            Dtype::FP64 => write!(f, "f64"),
            Dtype::BF16 => write!(f, "bf16"),
            Dtype::Pred => write!(f, "pred"),
        }
    }
}

impl From<&str> for Dtype {
    fn from(s: &str) -> Self {
        match s {
            "u8" => Dtype::U8,
            "s8" => Dtype::I8,
            "u16" => Dtype::U16,
            "s16" => Dtype::I16,
            "u32" => Dtype::U32,
            "s32" => Dtype::I32,
            "u64" => Dtype::U64,
            "s64" => Dtype::I64,
            "f16" => Dtype::FP16,
            "f32" => Dtype::FP32,
            "f64" => Dtype::FP64,
            "bf16" => Dtype::BF16,
            "pred" => Dtype::Pred,
            _ => panic!("Unknown dtype: {}", s),
        }
    }
}

impl serde::Serialize for Dtype {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(match self {
            Dtype::U8 => "jnp.uint8",
            Dtype::I8 => "jnp.int8",
            Dtype::U16 => "jnp.uint16",
            Dtype::I16 => "jnp.int16",
            Dtype::U32 => "jnp.uint32",
            Dtype::I32 => "jnp.int32",
            Dtype::U64 => "jnp.uint64",
            Dtype::I64 => "jnp.int64",
            Dtype::FP16 => "jnp.float16",
            Dtype::FP32 => "jnp.float32",
            Dtype::FP64 => "jnp.float64",
            Dtype::BF16 => "jnp.bfloat16",
            Dtype::Pred => "jnp.bool_",
        })
    }
}
