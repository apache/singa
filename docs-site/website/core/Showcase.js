/*
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
*/

/*
  from docusaurus1.x core/Showcase.js
  renders the users to index.js page
*/

const React = require("react")
const PropTypes = require("prop-types")

const UserLink = ({ infoLink, image, caption }) => (
  <a className="link" href={infoLink} key={infoLink}>
    <img src={image} alt={caption} title={caption} />
    <span className="caption">{caption}</span>
  </a>
)

UserLink.propTypes = {
  infoLink: PropTypes.string.isRequired,
  image: PropTypes.string.isRequired,
  caption: PropTypes.string.isRequired,
}

const Showcase = ({ users }) => (
  <div className="showcase">
    {users.map(user => (
      <UserLink key={user.infoLink} {...user} />
    ))}
  </div>
)

Showcase.propTypes = {
  users: PropTypes.array.isRequired,
}

module.exports = Showcase
