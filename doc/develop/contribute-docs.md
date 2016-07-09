## How to Contribute Documentation

___

This document gives step-by-step instructions for deploying [Singa website](http://singa.incubator.apache.org).

Singa website is managed by [Apache CMS](http://www.apache.org/dev/cms.html).

Singa website is built by [Maven](https://maven.apache.org) from a source tree stored in svn: https://svn.apache.org/repos/asf/incubator/singa/site/trunk.

### Edit Source Content

You can edit source content in 2 ways:

1. Use the CMS UI through your web browser:
	* Go to https://cms.apache.org/singa/ and install the bookmarklet.
	* Go to the webpage you want to modify.
	* Click the installed ASF CMS bookmarklet which enable you to browse the content in CMS.
	* Navigate to the content you want to modify and click the button "Edit".
	* Once you have modified the content, commit with the button "Submit".
2. Checkout with svn and work locally:
	* Checkout the source content: `svn co https://svn.apache.org/repos/asf/incubator/singa/site/trunk`.
	* Modify the content with your favorite text editor.
	* Test website in local: `mvn site`.
	* Check-in source modifications.

A video tutorial on how to use the CMS is available [here](http://s.apache.org/cms-tutorial).

If you are not a committer and want to edit the content, please see [here](http://www.apache.org/dev/cmsref.html#non-committer). A video tutorial for anonymous users is available [here](http://s.apache.org/cms-anonymous-tutorial).

After source tree is modified in svn, a Buildbot job is triggered:

1. It builds the HTML site using maven-site-plugin: `mvn site`.
2. It publishes generated HTML content to CMS [staging svn area](https://svn.apache.org/repos/infra/websites/staging/singa/trunk/content/).
3. Svnpubsub mechanism transfers svn CMS staging content to live CMS staging site: http://singa.staging.apache.org.


### Publish site content

If everything is good, publish modifications using [CMS publish](https://cms.apache.org/singa/publish) action.

Under the hood:

1. CMS copies CMS staging svn area content to [website production svn area](https://svn.apache.org/repos/infra/websites/production/singa/content/).
2. Svnpubsub mechanism transfers svn production content to live production site: http://singa.incubator.apache.org.
