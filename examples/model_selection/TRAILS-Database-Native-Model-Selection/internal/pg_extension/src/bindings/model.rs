/************************************************************
* 
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

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
