#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<algorithm>
using namespace std;

typedef unsigned char uchar;

typedef unsigned int uint;
typedef unsigned long long int uint64;

FILE *errout;
const uint64 C_MAX = 512;
const uint64 SIZE_TREE = 1024;
const uint64 BUF_MAX = 1024;
const uint64 POWER = 32;
const uint64 MAX_PROBABILITY = 1000000000;
uchar buf[BUF_MAX];
struct vertex {
    uint64 add;
    uint c;
    uint64 b;
    uint size;
};
vertex vx, vy, vret0 = {0, 0, 0};
struct probability_table {
    uint64 b[C_MAX];
    vertex t[C_MAX * 4];
    uint len;
    uint64 agr;
    probability_table()
    {
        for (int i = 0; i < C_MAX; i++) 
            b[i] = 1; 
    }
    
    void fill(FILE *f) 
    {
        fseek(f, 0, SEEK_SET);
        len = 0;
        while (!feof(f)) {
			len += fread(buf, sizeof(uchar), BUF_MAX, f); 
        }
        fseek(f, 0, SEEK_SET);
    }

   
    void print(FILE *f) 
    {
        fwrite(&len, sizeof(len), 1, f);
    }
    
    void read(FILE *f)
    {
        fread(&len, sizeof(len), 1, f);
    }

    void build_tree() {    
        int shift = SIZE_TREE / 2 - 1;
        for (int i = SIZE_TREE - 2; i >= shift; i--) {
            t[i].c = i - shift;
            t[i].add = 0;
            t[i].b = b[t[i].c];
            t[i].size = 1;
        }
        int left, right;
        for (int i = SIZE_TREE / 2 - 2; i >= 0; i--) {
            left = i * 2 + 1;
            right = i * 2 + 2;
            t[i].c = 0;
            t[i].b = t[left].b + t[right].b;
            t[i].size = t[left].size + t[right].size;
            t[i].add = 0;
        }
    };

    void add_tree(uint v, uint lv, uint rv, uint l, uint r, uint64 x){
        if (rv < l || lv > r) return;
        if (lv >= l && r >= rv) {
            t[v].add += x;
            return;
        }
        add_tree(v * 2 + 1, lv, (rv + lv) / 2, l, r, x);
        add_tree(v * 2 + 2, (rv + lv) / 2 + 1, rv, l, r, x);
        uint left = v * 2 + 1;
        uint right = v * 2 + 2;
        
        t[v].b = t[left].b + t[left].add * t[left].size + t[right].b + t[right].add * t[right].size;  
        t[v].c = 0;
    };

    uint64 tree_sum(uint v, uint lv, uint rv, uint l, uint r) { 
        if (rv < l || lv > r) {
            return 0;
        }
        if (lv >= l && r >= rv) {
            return t[v].add * t[v].size + t[v].b;
        }
        uint64 suml = tree_sum(v * 2 + 1, lv, (rv + lv) / 2, l, r);
        uint64 sumr = tree_sum(v * 2 + 2, (rv + lv) / 2 + 1, rv, l, r);
        uint common_len = min(r, rv) - max(l, lv) + 1;
        return suml + sumr + t[v].add * common_len;
    }
    
    void decrease_tree(uint v, uint64 add) 
    {
        t[v].b += (add + t[v].add) * t[v].size;
        t[v].b /= 2;
        if (t[v].b == 0) t[v].b = 1;
        if (v >= C_MAX - 1) {
            t[v].add = 0;
            return;
        }
        decrease_tree(2 * v + 1, t[v].add + add);
        decrease_tree(2 * v + 2, t[v].add + add);
        t[v].add = 0;
        t[v].b = t[2 * v + 1].b + t[2 * v + 2].b;
        t[v].c = 0;
    }
    
    uint64 get_probability(uint c) 
    {
	return tree_sum(0, 0, C_MAX - 1, 0, c);
    }

    void decrease_probability() 
    {
        decrease_tree(0, 0);
    }

    void add_probability(uint c)
    {
		add_tree(0, 0, C_MAX - 1, c, c, agr);
		while (t[0].b + t[0].add * t[0].size > MAX_PROBABILITY) decrease_probability();
    }
    
    uchar find(uint64 x, uint64 *new_l, uint64 *new_h, uint64 old_l, uint64 old_h)
    {
        uint v = 0, left, right;
        uint64 B = t[0].b + t[0].add * t[0].size;
        uint64 pref_sum = 0, l_sum, h_sum, l, m, h, parent_add = 0;
        while (v <= SIZE_TREE / 2 - 2) {
            left = v * 2 + 1;
            l_sum = t[left].b + (t[left].add + parent_add) * t[left].size;
            right = v * 2 + 2;    
            h_sum = t[right].b + (t[right].add + parent_add) * t[right].size;
            parent_add += t[v].add;
            l = old_l + pref_sum * (old_h - old_l + 1) / B;
            m = old_l + (pref_sum + l_sum) * (old_h - old_l + 1) / B - 1;
            h = old_l + (pref_sum + l_sum + h_sum) * (old_h - old_l + 1) / B - 1;
            if (m >= x) {
                v = left;
                *new_l = l;
                *new_h = m;
            }
            else {
                pref_sum += l_sum; 
                v = right;
                *new_l = m + 1;
                *new_h = h;
            }   
        }
        return t[v].c;
    }

    void scale(uint64 *new_l, uint64 *new_h, uint64 old_l, uint64 old_h, uint c) 
    {
        uint64 l = old_l;
        uint64 bc, bc_1, B;
        bc = get_probability(c);

        B = t[0].b + t[0].add * t[0].size;
        if (c != 0) {
			bc_1 = get_probability(c - 1);
            l = old_l + bc_1 * (old_h - old_l + 1) / B;
		}
        uint64 h = old_l + bc * (old_h - old_l + 1) / B - 1;
        *new_l = l;
        *new_h = h;
    }


        
} table;

FILE *input;
const uint64 BUF_INP_MAX = 1024;
uchar buf_inp[BUF_INP_MAX];

uint64 next_char(uchar *c)
{
    static uint64 cnt_c = 0, used_c = 0;
    
    if (cnt_c == 0 || used_c == cnt_c) { 
        cnt_c = fread(buf_inp, sizeof(uchar), BUF_INP_MAX, input);
        used_c = 0;
    }
    if (cnt_c == 0) return 0;
    *c = buf_inp[used_c++];
    return 1;
}
uint64 mask = (1ll << 32) - 1;
void next_bit(uint64 *x)
{
    static uint64 cnt_b = 0, used_b = 0;
    if (cnt_b == 0 || used_b == cnt_b) {
        cnt_b = fread(buf_inp, sizeof(uchar), BUF_INP_MAX, input) * 8ll;
        used_b = 0;
    }
    if (cnt_b == 0) {
        *x = (*x * 2ll);
        return;
    }
    uint64 shift = used_b % 8;
    uint64 np = (buf_inp[used_b / 8] >> shift) & 1ll;
    *x = ((*x << 1ll) | np);
    used_b++;
    return;
}

uchar buf_out = 0;
FILE *output;
uint64 MY_EOF = 0;

void next_print(uint64 x)
{
    static uint64 cnt2 = 0;
    if (cnt2 == 8) { 
        fwrite(&buf_out, sizeof(uchar), 1, output);
        cnt2 = 0;
        buf_out = 0;
    }
    buf_out |= x << cnt2;
    cnt2++;
    if (MY_EOF == 2 && buf_out != 0) {
        fwrite(&buf_out, sizeof(uchar), 1, output);
    }
}

void print_char(uchar c)
{
    fwrite(&c, sizeof(uchar), 1, output);
}
uint64 bits_to_follow = 0;

void bits_plus_follow(uint64 x)
{
    if (bits_to_follow == 0 && MY_EOF == 1) MY_EOF = 2;
    next_print(x);
    for (uint64 i = 0; i < bits_to_follow; i++) {
        if (i == bits_to_follow - 1 && MY_EOF == 1) MY_EOF = 2;
        next_print(1 - x);
    }
    bits_to_follow = 0;
}

const uint64 H = (1ll << POWER) - 1;
const uint64 L = 0;
const uint64 Half = (H + 1) / 2;
const uint64 Qtr1 = Half / 2;
const uint64 Qtr3 = Qtr1 * 3;
uint k = 0;
FILE *errin;

void compress()
{
    //errin = fopen("errin.txt", "w");
    table.fill(input);
    table.print(output);
    uint64 l = L, h = H;
    uchar c;
    while (next_char(&c)) {
        k++;
        table.scale(&l, &h, l, h, c);
        while (1) {
           //fprintf(errin, "%c(%u) %llu %llu\n", c, k, l, h);
	       if (h < Half) { 
                bits_plus_follow(0);
            }
            else if (l >= Half) {
                bits_plus_follow(1);
                l -= Half; 
                h -= Half;
            }
            else if (l > Qtr1 && h < Qtr3) {
                bits_to_follow++;
                l -= Qtr1;
                h -= Qtr1;
            } else break;
            l *= 2;
            h = h * 2 + 1;
        }
        table.add_probability(c);
    }
    MY_EOF = 1;
    if (l == 0) bits_plus_follow(0);
    else bits_plus_follow(1);
}
void decompress()
{
    //errout = fopen("errd.txt", "w");
    table.read(input);
    uint64 l = L, h = H;
    uint64 x = 0;
    uchar c;
    uint printed = 0;
    for (uint64 i = 0; i < POWER; i++) next_bit(&x);
    while (1) {
		if (printed == table.len) return;
        c = table.find(x, &l, &h, l, h);
        print_char(c);
        printed++;
        if (printed == table.len) return;
        while (1) { 
            //fprintf(errout, "%c(%u) %llu %llu %llu\n", c, printed, l, x, h);
			if (h < Half);
            else if (l >= Half) {
                x -= Half;
                l -= Half; 
                h -= Half;
            }
            else if (l > Qtr1 && h < Qtr3) {
                x -= Qtr1;
                l -= Qtr1;
                h -= Qtr1;
            } else break;
            l *= 2;
            h = h * 2 + 1;
            next_bit(&x);
        } 
        table.add_probability(c);
    }
}


int main(int argc, char *argv[])
{
    if (argc < 4) return -1;
    fopen_s(&input, argv[2], "rb");
    fopen_s(&output, argv[3], "wb");
    table.build_tree();
    table.agr = 1011010;
    if (argv[1][0] == 'c') compress();
    if (argv[1][0] == 'd') decompress();
    /* uint64 z = table.get_probability('a');
    z = table.get_probability('b');
    table.add_probability('a');
    z = table.get_probability('a');
    z = table.get_probability('b'); */
    return 0;
}
