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
#include "singa/singa_config.h"
#ifdef ENABLE_DIST
#include <assert.h>
#include <string.h>
#include <unistd.h>

#include <memory>

#include "singa/io/network.h"
#include "singa/utils/integer.h"
#include "singa/utils/logging.h"

#define SIZE 10000000
#define PORT 10000
#define ITER 10

using namespace singa;
int main(int argc, char **argv) {
  char *md = new char[SIZE];
  char *payload = new char[SIZE];

  const char *host = "localhost";
  int port = PORT;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-p") == 0)
      port = atoi(argv[++i]);
    else if (strcmp(argv[i], "-h") == 0)
      host = argv[++i];
    else
      fprintf(stderr, "Invalid option %s\n", argv[i]);
  }

  memset(md, 'a', SIZE);
  memset(payload, 'b', SIZE);

  NetworkThread *t = new NetworkThread(port);

  EndPointFactory *epf = t->epf_;

  // sleep
  sleep(3);

  EndPoint *ep = epf->getEp(host);

  Message *m[ITER];
  for (int i = 0; i < ITER; ++i) {
    m[i] = new Message();
    m[i]->setMetadata(md, SIZE);
    m[i]->setPayload(payload, SIZE);
  }

  while (1) {
    for (int i = 0; i < ITER; ++i) {
      if (ep->send(m[i]) < 0) return 1;
      delete m[i];
    }

    for (int i = 0; i < ITER; ++i) {
      m[i] = ep->recv();
      if (!m[i]) return 1;
      char *p;
      CHECK_EQ(m[i]->getMetadata((void **)&p), SIZE);
      CHECK_EQ(0, strncmp(p, md, SIZE));
      CHECK_EQ(m[i]->getPayload((void **)&p), SIZE);
      CHECK_EQ(0, strncmp(p, payload, SIZE));
    }
  }

  // while(ep && cnt++ <= 5 && ep->send(m) > 0 ) {

  //    LOG(INFO) << "Send a " << m->getSize() << " bytes message";

  //    Message* m1 = ep->recv();

  //    if (!m1)
  //        break;

  //    char *p;

  //    LOG(INFO) << "Receive a " << m1->getSize() << " bytes message";

  //    CHECK(m1->getMetadata((void**)&p) == SIZE);
  //    CHECK(0 == strncmp(p, md, SIZE));
  //    CHECK(m1->getPayload((void**)&p) == SIZE);
  //    CHECK(0 == strncmp(p, payload, SIZE));

  //    delete m;
  //    m = m1;
  //}
}
#endif  // ENABLE_DIST
