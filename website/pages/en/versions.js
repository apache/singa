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

/* versions page when click the version number
 next to the title in the header
*/

const React = require('react');

const CompLibrary = require('../../core/CompLibrary');

const Container = CompLibrary.Container;

const CWD = process.cwd();
// versions post docusaurus-upgrade Nov 2019
const versions = require(`${CWD}/versions.json`);
// versions pre docusaurus-upgrade Nov 2019
const oldversions = require(`${CWD}/oldversions.json`);

function Versions(props) {
  const {config: siteConfig} = props;
  const latestVersion = versions[0];
  const repoUrl = `https://github.com/${siteConfig.organizationName}/${siteConfig.projectName}`;
  return (
    <div className="docMainWrapper wrapper">
      <Container className="mainContainer versionsContainer">
        <div className="post">
          <header className="postHeader">
            <h1>{siteConfig.title} Versions</h1>
          </header>
          <h3 id="latest">Current version (Stable)</h3>
          <p>Current stable version of Apache SINGA</p>
          <table className="versions">
            <tbody>
              <tr>
                <th>{latestVersion}</th>
                <td>
                  <a
                    href={`${siteConfig.baseUrl}${siteConfig.docsUrl}/${
                      props.language ? props.language + '/' : ''
                    }installation`}>
                    Documentation
                  </a>
                </td>
                <td>
                  <a
                    href={`${siteConfig.baseUrl}${siteConfig.docsUrl}/${
                      props.language ? props.language + '/' : ''
                    }releases/RELEASE_NOTES_${latestVersion}.html`}>
                    Release Notes
                  </a>
                </td>
              </tr>
            </tbody>
          </table>
          <h3 id="rc">Pre-release versions</h3>
          <p>Latest unreleased <i>(next)</i> documentation and code of Apache SINGA</p>
          <table className="versions">
            <tbody>
              <tr>
                <th>master</th>
                <td>
                  <a
                    href={`${siteConfig.baseUrl}${siteConfig.docsUrl}/${
                      props.language ? props.language + '/' : ''
                    }next/installation`}>
                    Documentation
                  </a>
                </td>
                <td>
                  <a href={repoUrl}>Source Code</a>
                </td>
              </tr>
            </tbody>
          </table>
          <h3 id="archive">Past Versions</h3>
          <p>Here you can find previous versions of Apache SINGA</p>
          <p>Please refer to {' '}
            <a
              href={`${siteConfig.baseUrl}${siteConfig.docsUrl}/${
                props.language ? props.language + '/' : ''
              }download-singa#incubating-v010-8-october-2015`}>
              this page
            </a> for detailed release notes of previous versions.</p>
          <table className="versions">
            <tbody>
              {versions.map(
                version =>
                  version !== latestVersion && (
                    <tr key={version}>
                      <th>{version}</th>
                      <td>
                        <a
                          href={`${siteConfig.baseUrl}${siteConfig.docsUrl}/${
                            props.language ? props.language + '/' : ''
                          }${version}/installation`}>
                          Documentation
                        </a>
                      </td>
                      <td>
                        <a
                          href={`${siteConfig.baseUrl}${siteConfig.docsUrl}/${
                            props.language ? props.language + '/' : ''
                          }releases/RELEASE_NOTES_${version}.html`}>
                          Release Notes
                        </a>
                      </td>
                    </tr>
                  ),
              )}
              {oldversions.map(
                version =>
                  version !== latestVersion && (
                    <tr key={version}>
                      <th>{version}</th>
                      <td>
                        <a
                          href={`${siteConfig.baseUrl}v${version}/`}>                          Documentation
                        </a>
                      </td>
                      <td>
                        <a
                          href={`${siteConfig.baseUrl}${siteConfig.docsUrl}/${
                            props.language ? props.language + '/' : ''
                          }releases/RELEASE_NOTES_${version}.html`}>
                          Release Notes
                        </a>
                      </td>
                    </tr>
                  )
              )}
            </tbody>
          </table>
          <p>
            You can find past versions of this project on{' '}
            <a href={repoUrl}>GitHub</a>.
          </p>
        </div>
      </Container>
    </div>
  );
}

module.exports = Versions;
