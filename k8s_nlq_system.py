# Kubernetes Natural Language Query System with AWS Bedrock
# Requirements: pip install langgraph langchain kubernetes streamlit boto3 langchain-aws

import streamlit as st
import os
import boto3
from kubernetes import client, config
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import BedrockChat
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
import json
import re
import warnings
from dataclasses import dataclass
from enum import Enum

# Configuration
class QueryType(Enum):
    LIST_PODS = "list_pods"
    LIST_DEPLOYMENTS = "list_deployments" 
    LIST_SERVICES = "list_services"
    LIST_CONFIGMAPS = "list_configmaps"
    LIST_NAMESPACES = "list_namespaces"
    LIST_NODES = "list_nodes"
    UNSUPPORTED = "unsupported"

@dataclass
class QueryResult:
    success: bool
    data: Any = None
    warnings: List[str] = None
    error: str = None

# State definition for LangGraph
class QueryState(TypedDict):
    user_input: str
    parsed_query: Dict[str, Any]
    k8s_command: str
    result: QueryResult
    warnings: List[str]

class KubernetesNLProcessor:
    def __init__(self):
        # Initialize AWS Bedrock client
        self.bedrock_client = None
        self.llm = None
        
        try:
            # Initialize Bedrock client with credentials
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                region_name=os.getenv('AWS_REGION', 'us-east-1'),
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )
            
            # Initialize Bedrock Chat (using Claude-3 Haiku for cost efficiency)
            self.llm = BedrockChat(
                client=self.bedrock_client,
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                model_kwargs={
                    "max_tokens": 1000,
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            )
            self.bedrock_connected = True
            
        except Exception as e:
            self.bedrock_connected = False
            st.warning(f"Bedrock connection failed: {e}. Using rule-based parsing.")

        # Initialize Kubernetes client
        try:
            # Try in-cluster config first, then local config
            try:
                config.load_incluster_config()
            except:
                config.load_kube_config()
            
            self.v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
            self.k8s_connected = True
        except Exception as e:
            self.k8s_connected = False
            st.error(f"Failed to connect to Kubernetes cluster: {e}")

        # Sensitive data patterns
        self.sensitive_patterns = [
            r'password', r'secret', r'token', r'key', r'credential',
            r'cert', r'private', r'auth', r'api[_-]?key'
        ]
        
    def parse_with_bedrock(self, user_input: str) -> Dict[str, Any]:
        """Use Bedrock LLM to parse natural language query."""
        if not self.bedrock_connected or not self.llm:
            return self.parse_with_rules(user_input)
        
        try:
            prompt = f"""
            You are a Kubernetes query parser. Parse the following natural language query into a structured format.
            
            User Query: "{user_input}"
            
            Return ONLY a JSON object with these fields:
            - "type": one of ["list_pods", "list_deployments", "list_services", "list_configmaps", "list_namespaces", "list_nodes", "unsupported"]
            - "namespace": the namespace name or "default" or "all"
            
            Examples:
            - "list all pods" -> {{"type": "list_pods", "namespace": "all"}}
            - "show deployments in kube-system" -> {{"type": "list_deployments", "namespace": "kube-system"}}
            - "get services" -> {{"type": "list_services", "namespace": "default"}}
            
            JSON Response:
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Clean up the response and parse JSON
            response_text = response.content.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            parsed = json.loads(response_text)
            
            # Validate the response
            valid_types = ["list_pods", "list_deployments", "list_services", 
                          "list_configmaps", "list_namespaces", "list_nodes", "unsupported"]
            
            if parsed.get("type") not in valid_types:
                parsed["type"] = "unsupported"
            
            # Convert type string to enum
            type_mapping = {
                "list_pods": QueryType.LIST_PODS,
                "list_deployments": QueryType.LIST_DEPLOYMENTS,
                "list_services": QueryType.LIST_SERVICES,
                "list_configmaps": QueryType.LIST_CONFIGMAPS,
                "list_namespaces": QueryType.LIST_NAMESPACES,
                "list_nodes": QueryType.LIST_NODES,
                "unsupported": QueryType.UNSUPPORTED
            }
            
            parsed["type"] = type_mapping.get(parsed["type"], QueryType.UNSUPPORTED)
            
            return parsed
            
        except Exception as e:
            st.warning(f"Bedrock parsing failed: {e}. Falling back to rule-based parsing.")
            return self.parse_with_rules(user_input)
    
    def parse_with_rules(self, user_input: str) -> Dict[str, Any]:
        """Fallback rule-based parsing."""
        user_input = user_input.lower().strip()
        parsed_query = {"type": QueryType.UNSUPPORTED, "namespace": "default"}
        
        if "list" in user_input or "show" in user_input or "get" in user_input:
            if "pod" in user_input:
                parsed_query["type"] = QueryType.LIST_PODS
            elif "deployment" in user_input:
                parsed_query["type"] = QueryType.LIST_DEPLOYMENTS
            elif "service" in user_input:
                parsed_query["type"] = QueryType.LIST_SERVICES
            elif "configmap" in user_input or "config map" in user_input:
                parsed_query["type"] = QueryType.LIST_CONFIGMAPS
            elif "namespace" in user_input:
                parsed_query["type"] = QueryType.LIST_NAMESPACES
            elif "node" in user_input:
                parsed_query["type"] = QueryType.LIST_NODES
        
        # Extract namespace if specified
        namespace_match = re.search(r'namespace\s+(\w+)', user_input)
        if namespace_match:
            parsed_query["namespace"] = namespace_match.group(1)
        elif "all namespaces" in user_input or "all-namespaces" in user_input:
            parsed_query["namespace"] = "all"
            
        return parsed_query
        
    def parse_natural_language(self, state: QueryState) -> QueryState:
        """Parse natural language input to determine query type and parameters."""
        user_input = state["user_input"]
        
        # Use Bedrock if available, otherwise fall back to rules
        parsed_query = self.parse_with_bedrock(user_input)
        
        state["parsed_query"] = parsed_query
        return state
    
    def validate_query(self, state: QueryState) -> QueryState:
        """Validate the query and check for potential security issues."""
        warnings = []
        parsed_query = state["parsed_query"]
        
        # Check if query type is supported
        if parsed_query["type"] == QueryType.UNSUPPORTED:
            warnings.append("⚠️ Unsupported query type. Only listing operations are allowed.")
        
        # Check for sensitive data requests
        user_input = state["user_input"].lower()
        for pattern in self.sensitive_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                warnings.append(f"🔒 WARNING: Query may involve sensitive data ({pattern}). Proceed with caution.")
        
        # Special warnings for certain resource types
        if parsed_query["type"] == QueryType.LIST_CONFIGMAPS:
            warnings.append("🔐 ConfigMaps may contain sensitive configuration data.")
        
        state["warnings"] = warnings
        return state
    
    def execute_kubernetes_query(self, state: QueryState) -> QueryState:
        """Execute the Kubernetes query."""
        if not self.k8s_connected:
            state["result"] = QueryResult(
                success=False,
                error="Not connected to Kubernetes cluster"
            )
            return state
        
        parsed_query = state["parsed_query"]
        query_type = parsed_query["type"]
        namespace = parsed_query.get("namespace", "default")
        
        try:
            result_data = []
            
            if query_type == QueryType.LIST_PODS:
                if namespace == "all":
                    pods = self.v1.list_pod_for_all_namespaces()
                else:
                    pods = self.v1.list_namespaced_pod(namespace=namespace)
                
                result_data = [{
                    "name": pod.metadata.name,
                    "namespace": pod.metadata.namespace,
                    "status": pod.status.phase,
                    "node": pod.spec.node_name,
                    "ready": f"{sum(1 for c in (pod.status.container_statuses or []) if c.ready)}/{len(pod.spec.containers)}",
                    "restarts": sum(c.restart_count for c in (pod.status.container_statuses or [])),
                    "age": (pod.metadata.creation_timestamp).strftime('%Y-%m-%d %H:%M:%S') if pod.metadata.creation_timestamp else "Unknown"
                } for pod in pods.items]
                
            elif query_type == QueryType.LIST_DEPLOYMENTS:
                if namespace == "all":
                    deployments = self.apps_v1.list_deployment_for_all_namespaces()
                else:
                    deployments = self.apps_v1.list_namespaced_deployment(namespace=namespace)
                
                result_data = [{
                    "name": dep.metadata.name,
                    "namespace": dep.metadata.namespace,
                    "ready": f"{dep.status.ready_replicas or 0}/{dep.spec.replicas}",
                    "up_to_date": dep.status.updated_replicas or 0,
                    "available": dep.status.available_replicas or 0,
                    "age": (dep.metadata.creation_timestamp).strftime('%Y-%m-%d %H:%M:%S') if dep.metadata.creation_timestamp else "Unknown"
                } for dep in deployments.items]
                
            elif query_type == QueryType.LIST_SERVICES:
                if namespace == "all":
                    services = self.v1.list_service_for_all_namespaces()
                else:
                    services = self.v1.list_namespaced_service(namespace=namespace)
                
                result_data = [{
                    "name": svc.metadata.name,
                    "namespace": svc.metadata.namespace,
                    "type": svc.spec.type,
                    "cluster_ip": svc.spec.cluster_ip,
                    "external_ip": ', '.join(svc.status.load_balancer.ingress or []) if svc.status.load_balancer and svc.status.load_balancer.ingress else "<none>",
                    "ports": ', '.join([f"{port.port}:{port.target_port}/{port.protocol}" for port in (svc.spec.ports or [])]),
                    "age": (svc.metadata.creation_timestamp).strftime('%Y-%m-%d %H:%M:%S') if svc.metadata.creation_timestamp else "Unknown"
                } for svc in services.items]
                
            elif query_type == QueryType.LIST_CONFIGMAPS:
                if namespace == "all":
                    configmaps = self.v1.list_config_map_for_all_namespaces()
                else:
                    configmaps = self.v1.list_namespaced_config_map(namespace=namespace)
                
                result_data = [{
                    "name": cm.metadata.name,
                    "namespace": cm.metadata.namespace,
                    "data_keys": ', '.join(list(cm.data.keys())) if cm.data else "None",
                    "age": (cm.metadata.creation_timestamp).strftime('%Y-%m-%d %H:%M:%S') if cm.metadata.creation_timestamp else "Unknown"
                } for cm in configmaps.items]
                
            elif query_type == QueryType.LIST_NAMESPACES:
                namespaces = self.v1.list_namespace()
                result_data = [{
                    "name": ns.metadata.name,
                    "status": ns.status.phase,
                    "age": (ns.metadata.creation_timestamp).strftime('%Y-%m-%d %H:%M:%S') if ns.metadata.creation_timestamp else "Unknown"
                } for ns in namespaces.items]
                
            elif query_type == QueryType.LIST_NODES:
                nodes = self.v1.list_node()
                result_data = [{
                    "name": node.metadata.name,
                    "status": "Ready" if any(condition.type == "Ready" and condition.status == "True" 
                                           for condition in node.status.conditions) else "NotReady",
                    "roles": ', '.join([label.split('/')[-1] for label in node.metadata.labels.keys() 
                                      if 'node-role' in label]) or "worker",
                    "version": node.status.node_info.kubelet_version,
                    "internal_ip": next((addr.address for addr in node.status.addresses 
                                       if addr.type == "InternalIP"), "Unknown"),
                    "age": (node.metadata.creation_timestamp).strftime('%Y-%m-%d %H:%M:%S') if node.metadata.creation_timestamp else "Unknown"
                } for node in nodes.items]
            
            state["result"] = QueryResult(success=True, data=result_data)
            
        except Exception as e:
            state["result"] = QueryResult(
                success=False,
                error=f"Kubernetes API error: {str(e)}"
            )
            
        return state
    
    def create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow."""
        workflow = StateGraph(QueryState)
        
        # Add nodes
        workflow.add_node("parse", self.parse_natural_language)
        workflow.add_node("validate", self.validate_query)
        workflow.add_node("execute", self.execute_kubernetes_query)
        
        # Add edges
        workflow.add_edge("parse", "validate")
        workflow.add_edge("validate", "execute")
        workflow.add_edge("execute", END)
        
        # Set entry point
        workflow.set_entry_point("parse")
        
        return workflow.compile()

def main():
    st.set_page_config(
        page_title="K8s NL Query System (AWS Bedrock)",
        page_icon="☸️",
        layout="wide"
    )
    
    st.title("☸️ Kubernetes Natural Language Query System")
    st.markdown("*Powered by AWS Bedrock and LangGraph*")
    
    # Initialize the processor
    if 'processor' not in st.session_state:
        st.session_state.processor = KubernetesNLProcessor()
        st.session_state.workflow = st.session_state.processor.create_workflow()
    
    # Status indicators
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.processor.k8s_connected:
            st.success("✅ Connected to Kubernetes cluster")
        else:
            st.error("❌ Not connected to Kubernetes cluster")
    
    with col2:
        if st.session_state.processor.bedrock_connected:
            st.success("✅ Connected to AWS Bedrock")
        else:
            st.warning("⚠️ Using rule-based parsing (Bedrock unavailable)")
    
    # AWS Region info
    st.info(f"🌍 AWS Region: {os.getenv('AWS_REGION', 'us-east-1')}")
    
    # Input section
    st.subheader("Ask a Question")
    
    # Example queries
    with st.expander("📝 Example Queries"):
        st.write("""
        - "List all pods in the default namespace"
        - "Show me all deployments"
        - "Get services in kube-system namespace" 
        - "List configmaps in all namespaces"
        - "Show all namespaces"
        - "List all nodes"
        - "Get pods that are running"
        """)
    
    # Text input
    user_query = st.text_input(
        "Enter your query:",
        placeholder="e.g., 'List all pods in the default namespace'",
        help="Type your Kubernetes query in natural language"
    )
    
    if st.button("🔍 Execute Query", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a query!")
            return
            
        # Execute workflow
        with st.spinner("Processing query with AWS Bedrock..."):
            initial_state = QueryState(
                user_input=user_query,
                parsed_query={},
                k8s_command="",
                result=QueryResult(success=False),
                warnings=[]
            )
            
            final_state = st.session_state.workflow.invoke(initial_state)
        
        # Display parsed query info
        with st.expander("🧠 Query Analysis"):
            parsed = final_state["parsed_query"]
            st.write(f"**Query Type:** {parsed.get('type', 'Unknown')}")
            st.write(f"**Namespace:** {parsed.get('namespace', 'default')}")
            st.write(f"**Processing Method:** {'AWS Bedrock LLM' if st.session_state.processor.bedrock_connected else 'Rule-based'}")
        
        # Display warnings
        if final_state["warnings"]:
            for warning in final_state["warnings"]:
                st.warning(warning)
        
        # Display results
        result = final_state["result"]
        
        if result.success:
            st.success(f"✅ Query executed successfully!")
            
            if result.data:
                st.subheader("Results")
                
                # Display as dataframe for better visualization
                import pandas as pd
                df = pd.DataFrame(result.data)
                st.dataframe(df, use_container_width=True)
                
                # Show count
                st.info(f"Found {len(result.data)} items")
                
                # Download option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"k8s_query_results.csv",
                    mime="text/csv"
                )
                
                # JSON view option
                with st.expander("🔍 View Raw JSON"):
                    st.json(result.data)
            else:
                st.info("No data returned from query")
        else:
            st.error(f"❌ Query failed: {result.error}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Security Notes:**
    - Only listing operations are supported for safety
    - Queries involving sensitive data will show warnings  
    - ConfigMaps and Secrets require extra caution
    - All queries are processed through AWS Bedrock for enhanced understanding
    """)

if __name__ == "__main__":
    main()
