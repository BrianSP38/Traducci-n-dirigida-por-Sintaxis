import re
import sys
import math
import matplotlib.pyplot as plt

# Tokens
TOK_NUMBER = 'NUMBER'
TOK_PLUS   = 'PLUS'
TOK_MINUS  = 'MINUS'
TOK_MUL    = 'MUL'
TOK_DIV    = 'DIV'
TOK_LPAREN = 'LPAREN'
TOK_RPAREN = 'RPAREN'
TOK_ID     = 'ID'
TOK_ASSIGN = 'ASSIGN'
TOK_EOF    = 'EOF'

token_specification = [
    (TOK_NUMBER,  r'\d+(\.\d+)?'),  
    (TOK_ID,      r'[A-Za-z_][A-Za-z0-9_]*'),
    (TOK_PLUS,    r'\+'),
    (TOK_MINUS,   r'-'),
    (TOK_MUL,     r'\*'),
    (TOK_DIV,     r'/'),
    (TOK_LPAREN,  r'\('),
    (TOK_RPAREN,  r'\)'),
    (TOK_ASSIGN,  r'='),
    ('SKIP',      r'[ \t]+'),
    ('MISMATCH',  r'.'),   
]

master_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_specification)
get_token = re.compile(master_regex).match

class Token:
    def __init__(self, type_, value, pos):
        self.type = type_
        self.value = value
        self.pos = pos
    def __repr__(self):
        return f'Token({self.type}, {self.value})'

def tokenize(text):
    pos = 0
    mo = get_token(text, pos)
    tokens = []
    while mo is not None:
        kind = mo.lastgroup
        val = mo.group(kind)
        if kind == 'SKIP':
            pass
        elif kind == 'MISMATCH':
            raise SyntaxError(f'Unexpected char {val!r} at pos {pos}')
        else:
            tokens.append(Token(kind, val, pos))
        pos = mo.end()
        mo = get_token(text, pos)
    tokens.append(Token(TOK_EOF, '', pos))
    return tokens

# AST Nodes
class ASTNode:
    def __init__(self):
        self.value = None
    def eval(self, symtab):
        raise NotImplementedError()

class NumberNode(ASTNode):
    def __init__(self, number_text):
        super().__init__()
        self.text = number_text
    def eval(self, symtab):
        if self.value is None:
            if '.' in self.text:
                self.value = float(self.text)
            else:
                self.value = int(self.text)
        return self.value
    def __repr__(self):
        return f'Number({self.text})'

class VarNode(ASTNode):
    def __init__(self, name):
        super().__init__()
        self.name = name
    def eval(self, symtab):
        if self.name in symtab:
            self.value = symtab[self.name]
            return self.value
        else:
            raise NameError(f"Variable '{self.name}' no definida.")
    def __repr__(self):
        return f'Var({self.name})'

class BinOpNode(ASTNode):
    def __init__(self, op, left, right):
        super().__init__()
        self.op = op
        self.left = left
        self.right = right
    def eval(self, symtab):
        if self.value is None:
            l = self.left.eval(symtab)
            r = self.right.eval(symtab)
            if self.op == '+':
                self.value = l + r
            elif self.op == '-':
                self.value = l - r
            elif self.op == '*':
                self.value = l * r
            elif self.op == '/':
                if r == 0:
                    raise ZeroDivisionError('División por cero.')
                self.value = l / r
            else:
                raise ValueError('Operador desconocido ' + self.op)
        return self.value
    def __repr__(self):
        return f'BinOp({self.op}, {self.left}, {self.right})'

class AssignNode(ASTNode):
    def __init__(self, name, expr):
        super().__init__()
        self.name = name
        self.expr = expr
    def eval(self, symtab):
        val = self.expr.eval(symtab)
        symtab[self.name] = val
        self.value = val
        return val
    def __repr__(self):
        return f'Assign({self.name}, {self.expr})'

# Parser
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current = tokens[0]

    def advance(self):
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current = self.tokens[self.pos]
        else:
            self.current = Token(TOK_EOF, '', self.current.pos)

    def accept(self, token_type):
        if self.current.type == token_type:
            val = self.current.value
            self.advance()
            return val
        return None

    def expect(self, token_type):
        if self.current.type == token_type:
            val = self.current.value
            self.advance()
            return val
        raise SyntaxError(f"Se esperaba {token_type} en la posición {self.current.pos}, encontrado {self.current.type}.")

    # STAT -> ID '=' E | E
    def parse_stat(self):
        if self.current.type == TOK_ID and self._lookahead_is_assign():
            name = self.accept(TOK_ID)
            self.expect(TOK_ASSIGN)
            expr = self.parse_E()
            return AssignNode(name, expr)
        else:
            return self.parse_E()

    def _lookahead_is_assign(self):
        if self.pos + 1 < len(self.tokens):
            return self.tokens[self.pos + 1].type == TOK_ASSIGN
        return False

    # E -> T E'
    def parse_E(self):
        left = self.parse_T()
        return self.parse_E_prime(left)

    # E' -> (+|-) T E' | ε
    def parse_E_prime(self, left):
        while self.current.type in (TOK_PLUS, TOK_MINUS):
            if self.accept(TOK_PLUS) is not None:
                right = self.parse_T()
                left = BinOpNode('+', left, right)
            elif self.accept(TOK_MINUS) is not None:
                right = self.parse_T()
                left = BinOpNode('-', left, right)
        return left

    # T -> F T'
    def parse_T(self):
        left = self.parse_F()
        return self.parse_T_prime(left)

    # T' -> (*|/) F T' | ε
    def parse_T_prime(self, left):
        while self.current.type in (TOK_MUL, TOK_DIV):
            if self.accept(TOK_MUL) is not None:
                right = self.parse_F()
                left = BinOpNode('*', left, right)
            elif self.accept(TOK_DIV) is not None:
                right = self.parse_F()
                left = BinOpNode('/', left, right)
        return left

    # F -> (E) | num | id
    def parse_F(self):
        if self.accept(TOK_LPAREN) is not None:
            node = self.parse_E()
            self.expect(TOK_RPAREN)
            return node
        elif self.current.type == TOK_NUMBER:
            txt = self.accept(TOK_NUMBER)
            return NumberNode(txt)
        elif self.current.type == TOK_ID:
            name = self.accept(TOK_ID)
            return VarNode(name)
        else:
            raise SyntaxError(f'Factor inesperado {self.current} en pos {self.current.pos}')

# Pretty print AST
def pretty_print_ast(node, indent=0):
    spacer = '  ' * indent
    if isinstance(node, NumberNode):
        print(f"{spacer}Number: {node.text}  (value: {node.value})")
    elif isinstance(node, VarNode):
        print(f"{spacer}Var: {node.name}  (value: {node.value})")
    elif isinstance(node, BinOpNode):
        print(f"{spacer}BinOp: {node.op}  (value: {node.value})")
        pretty_print_ast(node.left, indent + 1)
        pretty_print_ast(node.right, indent + 1)
    elif isinstance(node, AssignNode):
        print(f"{spacer}Assign: {node.name}  (value: {node.value})")
        pretty_print_ast(node.expr, indent + 1)
    else:
        print(f"{spacer}Nodo desconocido: {node}")

# Dibujo
def assign_positions(node):
    """
    Devuelve un dict positions: node -> (x, y)
    Usa recorrido in-order para binarios. Para AssignNode coloca el hijo bajo él.
    """
    positions = {}
    x_counter = {'x': 0} 
    def inorder(n, depth):
        if isinstance(n, BinOpNode):
            inorder(n.left, depth + 1)
            x = x_counter['x']
            positions[n] = (x, -depth)
            x_counter['x'] += 1
            inorder(n.right, depth + 1)
        elif isinstance(n, AssignNode):
            
            x = x_counter['x']
            positions[n] = (x, -depth)
            x_counter['x'] += 1
            inorder(n.expr, depth + 1)
        else:
            
            x = x_counter['x']
            positions[n] = (x, -depth)
            x_counter['x'] += 1
    inorder(node, 0)
    return positions

def draw_tree(node, filename='ast.png', figsize=(10,6)):
    positions = assign_positions(node)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    def draw_edges(n):
        if isinstance(n, BinOpNode):
            parent_pos = positions[n]
            for child in (n.left, n.right):
                child_pos = positions[child]
                ax.plot([parent_pos[0], child_pos[0]], [parent_pos[1], child_pos[1]], linewidth=1)
                draw_edges(child)
        elif isinstance(n, AssignNode):
            parent_pos = positions[n]
            child = n.expr
            child_pos = positions[child]
            ax.plot([parent_pos[0], child_pos[0]], [parent_pos[1], child_pos[1]], linewidth=1)
            draw_edges(child)
        else:
            pass

    draw_edges(node)

    # dibujar nodos (círculos) y texto
    for n, (x, y) in positions.items():
        if isinstance(n, BinOpNode):
            label = n.op
        elif isinstance(n, NumberNode):
            label = n.text
        elif isinstance(n, VarNode):
            label = n.name
        elif isinstance(n, AssignNode):
            label = f"= {n.name}"
        else:
            label = str(n)
        circle_radius = 0.25
        circ = plt.Circle((x, y), circle_radius, fill=True, alpha=0.9, linewidth=1, edgecolor='k')
        ax.add_patch(circ)
        # text (centro)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, color='white' if isinstance(n, (BinOpNode, AssignNode)) else 'black')

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    if xs and ys:
        margin = 1
        ax.set_xlim(min(xs)-margin, max(xs)+margin)
        ax.set_ylim(min(ys)-margin, max(ys)+margin)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close(fig)
    return filename


# Run / demo
def run_input(text, symtab):
    tokens = tokenize(text)
    parser = Parser(tokens)
    ast = parser.parse_stat()
    try:
        val = ast.eval(symtab)
    except Exception as e:
        val = None
    return val, ast

def main_cli_expr(expr):
    symtab = {}
    val, ast = run_input(expr, symtab)
    print("AST textual:")
    pretty_print_ast(ast)
    print("Valor (si pudo evaluarse):", val)
    fname = draw_tree(ast, filename='ast.png')
    print(f"AST exportado a: {fname}")
    print("Tabla de símbolos:", symtab)

def repl():
    print("EDTS aritmética - REPL con export de AST a ast.png")
    print("Escribe expresión (ej: 4+5*2 o x = 3+4). 'exit' para salir.")
    symtab = {}
    while True:
        try:
            line = input(">> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line.lower() in ('salir', 'exit', 'quit'):
            break
        try:
            val, ast = run_input(line, symtab)
            print("\nAST (texto) y evaluación:")
            pretty_print_ast(ast)
            print("Valor (si evaluable):", val)
            fname = draw_tree(ast, filename='ast.png')
            print(f"AST exportado a: {fname}")
            print("Tabla de símbolos:", symtab)
        except SyntaxError as e:
            print("Error sintáctico:", e)
        except NameError as e:
            print("Error de nombre:", e)
        except ZeroDivisionError as e:
            print("Error en evaluación:", e)
        except Exception as e:
            print("Error:", e)
        print("-" * 40)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        expr = ' '.join(sys.argv[1:])
        main_cli_expr(expr)
    else:
        repl()
