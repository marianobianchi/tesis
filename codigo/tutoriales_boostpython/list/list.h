#ifndef __LISTAENLAZADA__
#define __LISTAENLAZADA__

#include <iostream>
using namespace std;


class Lista {
    public:
        Lista();
        ~Lista();
        void agregarAdelante(const int a);
        int iesimo(int i) const;
        ostream& toStream(ostream& os) const;
    
    private:
        struct Node {
            Node* next;
            int data;
        };
        
        Node* m_first;
        Node* m_last;
};

ostream& operator<<(ostream& out, const Lista& a) {
	return a.toStream(out);
}


Lista::Lista(){
    m_first = 0;
    m_last = 0;
}

Lista::~Lista(){
    Node* next = m_first;
    Node* del;
    
    while(next != 0){
        del = next;
        next = next->next;
        delete del;
    }
}

void Lista::agregarAdelante(const int a){
    Node* new_node = new Node();
    new_node->data = a;
    
    new_node->next = m_first;
    m_first = new_node;
    
    if(m_last == 0){
        m_last = new_node;
    }
}

int Lista::iesimo(int i) const {
    Node* next = m_first;
    while(i > 0){
        next = next->next;
        i--;
    }
    return next->data;
}

ostream& Lista::toStream(ostream& os) const {

    os << '[';
    
    Node* next_node = m_first;
    int data;
    
    while(next_node != 0){
        data = next_node->data;
        os << data;
        if(next_node->next != 0) os << ',';
        next_node = next_node->next;
    }
    os << ']';
    
    return os;
    
}


#endif // __LISTAENLAZADA__
