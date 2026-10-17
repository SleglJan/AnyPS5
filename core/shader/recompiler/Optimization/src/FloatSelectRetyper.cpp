#include "Optimization/FloatSelectRetyper.hpp"
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace ShaderRecompiler {

namespace {

constexpr std::uint32_t Unbounded = std::numeric_limits<std::uint32_t>::max() / 4u;

bool IsCandidate(const IrValue& value) {
    return value.Opcode() == IrOpcode::SelectU32 || (value.IsPhi() && value.Type() == IrType::U32);
}

std::size_t FirstValueOperand(const IrValue& value) {
    return value.IsPhi() ? 0u : 1u;
}

bool IsValueOperand(const IrValue& user, std::size_t operand) {
    return user.IsPhi() || operand != 0u;
}

bool IsExactFloatImmediate(std::uint32_t bits) {
    const std::uint32_t exponent = bits & 0x7f800000u;
    const std::uint32_t fraction = bits & 0x007fffffu;
    return fraction == 0u || (exponent != 0u && exponent != 0x7f800000u);
}

class CutGraph {
public:
    explicit CutGraph(std::size_t nodes) : adjacency(nodes), level(nodes), cursor(nodes) {}

    void AddEdge(std::size_t from, std::size_t to, std::uint32_t capacity) {
        if (capacity == 0u || from == to) return;
        adjacency[from].push_back(edges.size());
        edges.push_back({to, capacity});
        adjacency[to].push_back(edges.size());
        edges.push_back({from, 0u});
    }

    std::vector<bool> SourceSide(std::size_t source, std::size_t sink) {
        while (BuildLevels(source, sink)) {
            std::fill(cursor.begin(), cursor.end(), 0u);
            while (Augment(source, sink) != 0u) {}
        }
        std::vector<bool> reached(adjacency.size(), false);
        std::vector<std::size_t> stack{source};
        reached[source] = true;
        while (!stack.empty()) {
            const std::size_t node = stack.back();
            stack.pop_back();
            for (const std::size_t index : adjacency[node]) {
                const Edge& edge = edges[index];
                if (edge.capacity == 0u || reached[edge.to]) continue;
                reached[edge.to] = true;
                stack.push_back(edge.to);
            }
        }
        return reached;
    }

private:
    struct Edge {
        std::size_t to;
        std::uint32_t capacity;
    };

    struct Step {
        std::size_t node;
        std::size_t edge;
    };

    bool BuildLevels(std::size_t source, std::size_t sink) {
        std::fill(level.begin(), level.end(), -1);
        std::vector<std::size_t> queue{source};
        level[source] = 0;
        for (std::size_t head = 0; head < queue.size(); ++head) {
            const std::size_t node = queue[head];
            for (const std::size_t index : adjacency[node]) {
                const Edge& edge = edges[index];
                if (edge.capacity == 0u || level[edge.to] >= 0) continue;
                level[edge.to] = level[node] + 1;
                queue.push_back(edge.to);
            }
        }
        return level[sink] >= 0;
    }

    std::uint32_t Augment(std::size_t source, std::size_t sink) {
        std::vector<Step> path;
        std::size_t node = source;
        while (true) {
            if (node == sink) {
                std::uint32_t pushed = Unbounded;
                for (const Step& step : path) pushed = std::min(pushed, edges[step.edge].capacity);
                for (const Step& step : path) {
                    edges[step.edge].capacity -= pushed;
                    edges[step.edge ^ 1u].capacity += pushed;
                }
                return pushed;
            }
            bool advanced = false;
            for (; cursor[node] < adjacency[node].size(); ++cursor[node]) {
                const std::size_t index = adjacency[node][cursor[node]];
                const Edge& edge = edges[index];
                if (edge.capacity == 0u || level[edge.to] != level[node] + 1) continue;
                path.push_back({node, index});
                node = edge.to;
                advanced = true;
                break;
            }
            if (advanced) continue;
            level[node] = -1;
            if (path.empty()) return 0u;
            node = path.back().node;
            path.pop_back();
            ++cursor[node];
        }
    }

    std::vector<Edge> edges;
    std::vector<std::vector<std::size_t>> adjacency;
    std::vector<int> level;
    std::vector<std::size_t> cursor;
};

void InsertAfterDefinition(IrValue& definition, IrValue& value) {
    IrBlock* block = definition.Parent();
    auto& instructions = block->Instructions();
    auto position = instructions.begin();
    while (position != instructions.end() && *position != &definition) ++position;
    if (position != instructions.end()) ++position;
    while (position != instructions.end() && (*position)->IsPhi()) ++position;
    if (position == instructions.end()) {
        block->AppendInstruction(&value);
        return;
    }
    block->InsertInstructionBefore(*position, &value);
}

}

FloatSelectRetypeStats FloatSelectRetyper::Retype(IrProgram& program) const {
    FloatSelectRetypeStats stats;
    std::vector<IrValue*> candidates;
    std::unordered_map<const IrValue*, std::size_t> node;
    for (IrBlock* block : program.BlockOrder()) {
        for (IrValue* inst : block->Instructions()) {
            if (!IsCandidate(*inst) || !inst->HasUses()) continue;
            node.emplace(inst, candidates.size());
            candidates.push_back(inst);
        }
    }
    if (candidates.empty()) return stats;
    const std::size_t source = candidates.size();
    const std::size_t sink = source + 1u;
    CutGraph graph(candidates.size() + 2u);
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        IrValue* value = candidates[index];
        std::uint32_t views = 0;
        bool integerUse = false;
        for (const auto& use : value->OperandUses()) {
            if (use.user->Opcode() == IrOpcode::BitCastF32U32) ++views;
            else if (!node.contains(use.user) || !IsValueOperand(*use.user, use.operand)) integerUse = true;
        }
        graph.AddEdge(source, index, views);
        if (integerUse) graph.AddEdge(index, sink, 1u);
        for (std::size_t operand = FirstValueOperand(*value); operand < value->ArgumentCount(); ++operand) {
            IrValue* input = value->Argument(operand)->Resolve();
            if (const auto found = node.find(input); found != node.end()) {
                graph.AddEdge(found->second, index, 1u);
                graph.AddEdge(index, found->second, 1u);
            } else if (input->Opcode() == IrOpcode::BitCastU32F32) {
                if (input->UseCount() == 1u) graph.AddEdge(source, index, 1u);
            } else if (input->HasImmediate() && input->Type() == IrType::U32) {
                if (!IsExactFloatImmediate(input->ImmediateU32())) graph.AddEdge(index, sink, Unbounded);
            } else if (input->Parent() != nullptr && input->Type() == IrType::U32) {
                graph.AddEdge(index, sink, 1u);
            } else {
                graph.AddEdge(index, sink, Unbounded);
            }
        }
    }
    const std::vector<bool> floatSide = graph.SourceSide(source, sink);
    std::vector<std::pair<IrValue*, IrValue*>> retypes;
    std::unordered_map<const IrValue*, IrValue*> floats;
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        if (!floatSide[index]) continue;
        IrValue* value = candidates[index];
        IrValue& retyped = program.CreateValue(value->IsPhi() ? IrOpcode::Phi : IrOpcode::SelectF32, IrType::F32);
        value->Parent()->InsertInstructionBefore(value, &retyped);
        floats.emplace(value, &retyped);
        retypes.emplace_back(value, &retyped);
    }
    if (retypes.empty()) return stats;
    std::unordered_map<const IrValue*, IrValue*> constants;
    const auto floatOf = [&](IrValue* input, IrBlock* block, IrValue* before) -> IrValue* {
        IrValue* resolved = input->Resolve();
        if (const auto found = floats.find(resolved); found != floats.end()) return found->second;
        if (resolved->Opcode() == IrOpcode::BitCastU32F32) return resolved->Argument(0);
        if (resolved->HasImmediate()) {
            if (const auto found = constants.find(resolved); found != constants.end()) return found->second;
            IrValue& constant = program.CreateValue(IrOpcode::Void, IrType::F32);
            constant.SetImmediateF32(std::bit_cast<float>(resolved->ImmediateU32()));
            constants.emplace(resolved, &constant);
            return &constant;
        }
        IrValue& view = program.CreateValue(IrOpcode::BitCastF32U32, IrType::F32);
        view.AddArgument(resolved);
        if (before != nullptr) block->InsertInstructionBefore(before, &view);
        else block->AppendInstruction(&view);
        return &view;
    };
    for (const auto& [value, retyped] : retypes) {
        if (value->IsPhi()) {
            for (std::size_t index = 0; index < value->ArgumentCount(); ++index) retyped->AddPhiOperand(value->PhiBlock(index), floatOf(value->Argument(index), value->PhiBlock(index), nullptr));
        } else {
            retyped->AddArgument(value->Argument(0));
            retyped->AddArgument(floatOf(value->Argument(1), retyped->Parent(), retyped));
            retyped->AddArgument(floatOf(value->Argument(2), retyped->Parent(), retyped));
        }
    }
    for (const auto& [value, retyped] : retypes) {
        for (std::size_t index = 0; index < value->ArgumentCount(); ++index) value->ReplaceArgument(index, nullptr);
    }
    for (const auto& [value, retyped] : retypes) {
        std::vector<IrValue*> viewers;
        std::vector<IrUse> integerUses;
        for (const auto& use : value->OperandUses()) {
            if (use.user->Opcode() == IrOpcode::BitCastF32U32) viewers.push_back(use.user);
            else integerUses.push_back(use);
        }
        for (IrValue* viewer : viewers) viewer->ReplaceAllUsesWith(retyped);
        if (!integerUses.empty()) {
            IrValue& back = program.CreateValue(IrOpcode::BitCastU32F32, IrType::U32);
            back.AddArgument(retyped);
            if (value->IsPhi()) InsertAfterDefinition(*value, back);
            else value->Parent()->InsertInstructionBefore(value, &back);
            for (const IrUse& use : integerUses) use.user->ReplaceArgument(use.operand, &back);
        }
        ++stats.retypedValues;
    }
    return stats;
}

}
