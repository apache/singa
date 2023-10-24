
use serde::{Serialize, Deserialize};


#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Frappe {
    pub(crate) id: i32,
    pub(crate) label: i32,
    pub(crate) col1: String,
    pub(crate) col2: String,
    pub(crate) col3: String,
    pub(crate) col4: String,
    pub(crate) col5: String,
    pub(crate) col6: String,
    pub(crate) col7: String,
    pub(crate) col8: String,
    pub(crate) col9: String,
    pub(crate) col10: String,
}
