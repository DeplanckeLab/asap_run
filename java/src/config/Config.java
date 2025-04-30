package config;

import json.ErrorJSON;

public class Config 
{
	private String dbHost = "postgres:5434"; // Default local host and port
	public static final String driver = "org.postgresql.Driver";

	private Config() {}

	private Config(String dbHost)
	{
		this.dbHost = dbHost;
	}

	public static Config ConfigMAIN()
	{
		return new Config();
	}

	public static Config ConfigDEV()
	{
		return new Config("updeplasrv4-new.epfl.ch:5433"); // Access via intranet
	}

	public String getHost()
	{
		return this.dbHost;
	}
	
	private String getEnv(String key, boolean required)
	{
		String value = System.getenv(key);
		if (required && (value == null || value.isEmpty())) new ErrorJSON("Missing required environment variable: " + key);
		return value;
	}

	public String getURL(String dbName)
	{
		String user = getEnv("POSTGRES_USER", true);
		String password = getEnv("POSTGRES_PASSWORD", true);
		return "jdbc:postgresql://" + this.dbHost + "/" + dbName + "?user=" + user + "&password=" + password;
	}

	public String getURLFromHost(String dbHostName)
	{
		String user = getEnv("POSTGRES_USER", true);
		String password = getEnv("POSTGRES_PASSWORD", true);
		return "jdbc:postgresql://" + dbHostName + "?user=" + user + "&password=" + password;
	}
}