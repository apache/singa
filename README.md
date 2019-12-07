# Apache SINGA docs-site

## yarn package name

`apache-singa-docs-site`

website workspace name: `apache-singa-docs-site-website-subdir`

refer to this page for writing this README: https://github.com/apache/pulsar/tree/master/site2

# Start Development Server

```sh
yarn run start:website
```

# Editing Content

## Editing an existing docs page

Edit docs by navigating to `docs/` and editing the corresponding document:

`docs/doc-to-be-edited.md`

```markdown
---
id: page-needs-edit
title: This Doc Needs To Be Edited
---

Edit me...
```

For more information about docs, click [here](https://docusaurus.io/docs/en/navigation)

## Editing an existing news post

Edit blog posts by navigating to `website/blog` and editing the corresponding post:

`website/blog/post-to-be-edited.md`

```markdown
---
id: post-needs-edit
title: This Blog Post Needs To Be Edited
---

Edit me...
```

For more information about blog posts, click [here](https://docusaurus.io/docs/en/adding-blog)

# Adding Content

## Adding a new docs page to an existing sidebar

1. Create the doc as a new markdown file in `/docs`, example `docs/newly-created-doc.md`:

```md
---
id: newly-created-doc
title: This Doc Needs To Be Edited
---

My new content here..
```

1. Refer to that doc's ID in an existing sidebar in `website/sidebar.json`:

```javascript
// Add newly-created-doc to the Getting Started category of docs
{
  "docs": {
    "Getting Started": [
      "quick-start",
      "newly-created-doc" // new doc here
    ],
    ...
  },
  ...
}
```

For more information about adding new docs, click [here](https://docusaurus.io/docs/en/navigation)

## Adding a new blog/news post

1. Make sure there is a header link to your blog in `website/siteConfig.js`:

`website/siteConfig.js`

```javascript
headerLinks: [
    ...
    { blog: true, label: 'Blog' },
    ...
]
```

2. Create the blog post with the format `YYYY-MM-DD-My-Blog-Post-Title.md` in `website/blog`:

`website/blog/2018-05-21-New-Blog-Post.md`

```markdown
---
author: Frank Li
authorURL: https://twitter.com/foobarbaz
authorFBID: 503283835
title: New Blog Post
---

Lorem Ipsum...
```

For more information about blog posts, click [here](https://docusaurus.io/docs/en/adding-blog)

## Adding items to your site's top navigation bar

1. Add links to docs, custom pages or external links by editing the headerLinks field of `website/siteConfig.js`:

`website/siteConfig.js`

```javascript
{
  headerLinks: [
    ...
    /* you can add docs */
    { doc: 'my-examples', label: 'Examples' },
    /* you can add custom pages */
    { page: 'help', label: 'Help' },
    /* you can add external links */
    { href: 'https://github.com/facebook/docusaurus', label: 'GitHub' },
    ...
  ],
  ...
}
```

For more information about the navigation bar, click [here](https://docusaurus.io/docs/en/navigation)

## Adding custom pages

1. Docusaurus uses React components to build pages. The components are saved as .js files in `website/pages/en`:
1. If you want your page to show up in your navigation header, you will need to update `website/siteConfig.js` to add to the `headerLinks` element:

`website/siteConfig.js`

```javascript
{
  headerLinks: [
    ...
    { page: 'my-new-custom-page', label: 'My New Custom Page' },
    ...
  ],
  ...
}
```

For more information about custom pages, click [here](https://docusaurus.io/docs/en/custom-pages).

# Full Documentation

Full documentation can be found on the [website](https://docusaurus.io/).

=== from Jest

#### Additional Workflow for any changes made to website or docs

If you are making changes to the website or documentation, test the website folder and run the server to check if your changes are being displayed accurately.

1.  Locate to the website directory and install any website specific dependencies by typing in `yarn`. Following steps are to be followed for this purpose from the root directory.
    ```sh-session
    $ cd website       # Only needed if you are not already in the website directory
    $ yarn
    $ yarn start
    ```
1.  You can run a development server to check if the changes you made are being displayed accurately by running `yarn start` in the website directory.

The Jest website also offers documentation for older versions of Jest, which you can edit in `website/versioned_docs`. After making changes to the current documentation in `docs`, please check if any older versions of the documentation have a copy of the file where the change is also relevant and apply the changes to the `versioned_docs` as well.

=== from pulsar

=== general gotchas

- cd to /website/build/singa/ to serve
- 1.1.0 and 0.3 old versions in singa-site DO NOT have translations (other than v0.3.0/zh/docs/index.html)
- TODO: move the singa-site folders to the build folders
- Static assets used in documents and blogs should go into docs/assets and website/blog/assets

for docusaurus 1x console log in react components will only show in live dev server terminal
