#include "singa/io/network/endpoint.h"
#include "singa/io/network/integer.h"
#include "singa/io/network/message.h"
#include <assert.h>
#include <unistd.h>

#include "singa/utils/logging.h"

#define SIZE 100
#define PORT 10000

using namespace singa;
int main(int argc, char** argv) {
    char md[SIZE];
    char payload[SIZE];

    char* host = "localhost";
    int port = PORT;

    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "-p") == 0)
            port = atoi(argv[++i]);
        else if (strcmp(argv[i], "-h") == 0)
            host = argv[++i];
        else
            fprintf(stderr, "Invalid option %s\n", argv[i]);
    }

    memset(md, 'a', SIZE);
    memset(payload, 'b', SIZE);

    Message* m = new Message();
    m->setMetadata(md, SIZE);
    m->setPayload(payload, SIZE);

    NetworkThread* t = new NetworkThread(port);

    EndPointFactory* epf = t->epf_;

    // sleep
    sleep(3);

    EndPoint* ep = epf->getEp(host);

    int cnt = 0;

    while(ep && cnt++ <= 100 && ep->send(m) > 0 ) {

        LOG(INFO) << "Send a " << m->getSize() << " bytes message";

        Message* m1 = ep->recv();

        if (!m1)
            break;

        char *p;

        LOG(INFO) << "Receive a " << m1->getSize() << " bytes message";

        CHECK(m1->getMetadata((void**)&p) == SIZE);
        CHECK(0 == strncmp(p, md, SIZE));
        CHECK(m1->getPayload((void**)&p) == SIZE);
        CHECK(0 == strncmp(p, payload, SIZE));

        delete m;
        m = m1;
    }
}
