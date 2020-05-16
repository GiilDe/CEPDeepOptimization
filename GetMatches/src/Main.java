import com.espertech.esper.common.client.EPCompiled;
import com.espertech.esper.common.client.configuration.Configuration;
import com.espertech.esper.common.internal.event.bean.core.BeanEventBean;
import com.espertech.esper.compiler.client.CompilerArguments;
import com.espertech.esper.compiler.client.EPCompileException;
import com.espertech.esper.compiler.client.EPCompiler;
import com.espertech.esper.compiler.client.EPCompilerProvider;
import com.espertech.esper.runtime.client.*;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import java.io.*;
import java.util.*;

import org.apache.log4j.BasicConfigurator;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;


public class Main {

    public static final String PATTERN = "GetMatches/pattern";

    static int toPrintTime = 10000;

    public static JSONObject CONSTANTS;

    public static final String JSON_PATH = "Data/constants.json";

    final static String STATEMENT_NAME = "s";


    static {
        JSONParser parser = new JSONParser();
        try {
            CONSTANTS = (JSONObject) parser.parse(new FileReader(JSON_PATH));
        } catch (IOException | ParseException e) {
            e.printStackTrace();
        }
    }

    static int MATCH_SIZE = Integer.parseInt((String) CONSTANTS.get("match_size"));

    public static final EsperRuntimeConfg ESPER_RUNTIME_CONFG = new EsperRuntimeConfg().invoke();
    public static final Configuration CONFIGURATION = ESPER_RUNTIME_CONFG.getConfiguration();
    public static final EPCompiled EP_COMPILED = ESPER_RUNTIME_CONFG.getEpCompiled();

    static List< ArrayList<Integer> > currentMatches = new LinkedList<>();

    public static UpdateListener matchMemorizer = (newData, oldData, s, r) -> {
        for(var match : newData){
            Object[] events = ((HashMap<String, BeanEventBean>)match.getUnderlying()).values().toArray();
            ArrayList<Integer> eventsCounts = new ArrayList<>(Collections.nCopies(MATCH_SIZE, 0));
            for (int i = 0; i < MATCH_SIZE; i++) {
                eventsCounts.set(i, (Integer) ((BeanEventBean)events[i]).get("count"));
            }
            currentMatches.add(eventsCounts);
        }
    };


    public static final Event DUMMY_EVENT = new Event("-1", -1, -1);

    public static void main(String[] s) {
        BasicConfigurator.configure();
        try {
            writeWindowsMatches((String) CONSTANTS.get("test_stream_path"),
                    (String) CONSTANTS.get("test_matches"));
            writeWindowsMatches((String) CONSTANTS.get("train_stream_path"),
                    (String) CONSTANTS.get("train_matches"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    static void writeWindowsMatches(String inputPath, String matchesPath) throws IOException {
        CSVParser eventsParser = CSVParser.parse(new FileReader(inputPath), CSVFormat.DEFAULT);
        CSVPrinter matchesPrinter = new CSVPrinter(new FileWriter(matchesPath), CSVFormat.EXCEL);

        EPRuntime runtime = getEpRuntime(matchMemorizer);

        long startTime = System.currentTimeMillis();

        Iterator<CSVRecord> events = eventsParser.iterator();

        final int windowSize = Integer.parseInt((String) CONSTANTS.get("window_size"));
        boolean finished = false;
        int progress = 0;
        while (true) {
            Event[] windowEvents = new Event[windowSize];

            int count = 0;
            for (int i = 0; i < windowSize; i++) {
                if (!events.hasNext()) {
                    finished = true;
                    break;
                }

                final CSVRecord nextEventRecord = events.next();

                final var nextEvent =
                        new Event(nextEventRecord.get(0), Double.parseDouble(nextEventRecord.get(1)), count);

                windowEvents[i] = nextEvent;
                count += 1;
            }

            if(finished) {
                break;
            }

            for (int i = 0; i < windowSize; i++) {
                final Event nextEvent = windowEvents[i];
                runtime.getEventService().sendEventBean(nextEvent, "Event");
            }

            if(currentMatches.isEmpty()){
                matchesPrinter.printRecord("-1");
            }
            else{
                for(var match : currentMatches){
                    for(Integer c : match){
                        matchesPrinter.print(c);
                    }
                }
                matchesPrinter.println();
            }

            currentMatches.clear();
            partition(runtime);

            progress += 1;
            printProgress(startTime, progress);
        }

        currentMatches.clear();
        partition(runtime);
        runtime.destroy();
        eventsParser.close();
        matchesPrinter.close();
    }

    private static void partition(EPRuntime runtime){
        final int patternWindowSize = Integer.parseInt((String) CONSTANTS.get("pattern_window_size"));
        for (int i = 0; i < patternWindowSize; i++) {
            runtime.getEventService().sendEventBean(DUMMY_EVENT, "Event");
        }
    }

    private static void printProgress(long startTime, int count) {
        if(count % 1000 == 0){
            System.out.println(count);
        }

        if(count % toPrintTime == 0){
            long currentTime = System.currentTimeMillis();
            double secPassed = ((double)(currentTime - startTime))/1000;
            double minPassed = secPassed/60;
            System.out.println("Time passed: " + secPassed + " secs");
            System.out.println("Time passed: " + minPassed + " mins");
        }
    }

    private static EPRuntime getEpRuntime(UpdateListener l) {
        EPRuntime runtime = EPRuntimeProvider.getDefaultRuntime(CONFIGURATION);
        EPDeployment deployment;
        try {
            deployment = runtime.getDeploymentService().deploy(EP_COMPILED);
        }
        catch (EPDeployException ex) {
            throw new RuntimeException(ex);
        }

        runtime.getDeploymentService().getStatement(deployment.getDeploymentId(),
                STATEMENT_NAME).addListener(l);

        return runtime;
    }

    private static class EsperRuntimeConfg {
        private Configuration configuration;
        private EPCompiled epCompiled;

        private static String getPattern() {
            StringBuilder pattern = new StringBuilder();
            BufferedReader reader;
            try {
                reader = new BufferedReader(new FileReader(PATTERN));
                String firstLine = reader.readLine();
                pattern.append(firstLine);
                String line = reader.readLine();
                while (line != null && !line.contentEquals("done")) {
                    pattern.append(" ").append(line);
                    line = reader.readLine();
                }
                reader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
            return pattern.toString();
        }

        public Configuration getConfiguration() {
            return configuration;
        }

        public EPCompiled getEpCompiled() {
            return epCompiled;
        }

        public EsperRuntimeConfg invoke() {
            EPCompiler compiler = EPCompilerProvider.getCompiler();
            configuration = new Configuration();
            configuration.getCommon().addEventType(Event.class);
            CompilerArguments args = new CompilerArguments(configuration);

            String pattern;
            try {
                pattern = getPattern();
                epCompiled = compiler.compile("@name('" + STATEMENT_NAME + "') " + pattern, args);
            }
            catch (EPCompileException ex) {
                throw new RuntimeException(ex);
            }
            return this;
        }
    }
}
