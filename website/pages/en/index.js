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
 Pages: The website/pages directory
 contains example top-level pages for the site.
 index.js is the landing page
*/

const React = require('react');

const CompLibrary = require('../../core/CompLibrary.js');

const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;
const Showcase = require(`${process.cwd()}/core/Showcase.js`);
// TODO: add <translate> tags
// const translate = require('../../server/translate.js').translate;

const siteConfig = require(`${process.cwd()}/siteConfig.js`);

function docUrl(doc, language) {
  return siteConfig.baseUrl + 'docs/' + (language ? language + '/' : '') + doc;
}

function HomeSplash(props) {
  const {siteConfig, language} = props;

  return (
    <div className="index-hero">
      {/* Increase the network loading priority of the background image. */}
      <img
        style={{ display: "none" }}
        src={`${siteConfig.baseUrl}img/unsplash-ocean.jpg`}
        alt="increase priority"
      />
      <div className="index-hero-inner">
        <img
          alt="SINGA-Ocean"
          className="index-hero-logo"
          src={`${siteConfig.baseUrl}img/singa.png`}
        />
        <h1 className="index-hero-project-tagline">
          A Distributed Deep Learning Platform
        </h1>
        <div className="index-ctas">
          <a
            className="button index-ctas-get-started-button"
            href={`${docUrl("installation", language)}`}>
            Get Started
          </a>
          <span className="index-ctas-github-button">
            <iframe
              src="https://ghbtns.com/github-btn.html?user=apache&amp;repo=singa&amp;type=star&amp;count=true&amp;size=large"
              frameBorder={0}
              scrolling={0}
              width={160}
              height={30}
              title="GitHub Stars"
            />
          </span>
        </div>
      </div>
    </div>
  );
}

class Index extends React.Component {
  render() {
    const {config: siteConfig, language = 'en'} = this.props;
    const pinnedUsersToShowcase = siteConfig.users.filter(user => user.pinned);

    return (
      <div>
        <HomeSplash siteConfig={siteConfig} language={language} />
        <div className="announcement">
          <div className="announcement-inner">
           {`Apache SINGA is an Apache Top Level Project, open source distributed training platform
            for deep learning and machine learning models`}
          </div>
        </div>
        <div className="mainContainer">
          <Container padding={['bottom', 'top']} background="light">
            <GridBlock
              align="center"
              contents={[
                {
                  content: `Apache SINGA focused on [distributed](https://dl.acm.org/citation.cfm?doid=2733373.2807410)
                   deep learning by partitioning the model and data onto nodes in a cluster and
                    [parallelize](https://dl.acm.org/citation.cfm?doid=2733373.2806232) the training`,
                  image: `${siteConfig.baseUrl}img/undraw_mind_map_cwng.svg`,
                  imageAlign: 'top',
                  imageAlt: 'Distributed Learning',
                  title: 'Distributed Learning',
                },
                {
                  content: `Apache SINGA v2.0.0 has AutoML features, a Healthcare
                   [model zoo](${docUrl("model-zoo-cnn-cifar10", language)}),
                    and facility for porting other models onto SINGA`,
                  image: `${siteConfig.baseUrl}img/undraw_web_development_w2vv.svg`,
                  imageAlign: 'top',
                  imageAlt: 'AutoML and Model Zoo',
                  title: 'AutoML and Model Zoo',
                },
              ]}
              layout="twoColumn"
            />
          </Container>
          <div className="productShowcaseSection paddingBottom">
            <h2 style={{ color: "#904600" }}>
              Who is Using Apache SINGA?
            </h2>
            <p>
              Apache SINGA powers many organizations and companies...
            </p>
            <Showcase users={pinnedUsersToShowcase} />
            <div className="more-users">
              <a
                className="button"
                href={`${siteConfig.baseUrl}${this.props.language}/users`}>
                All Apache SINGA Users
              </a>
            </div>
          </div>
        </div>
      </div>
    );
  }
}

module.exports = Index;
